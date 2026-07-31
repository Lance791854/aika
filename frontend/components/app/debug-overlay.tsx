'use client';

import { useEffect, useMemo, useState } from 'react';
import { useDataChannel } from '@livekit/components-react';

type ActiveTimer = { name: string; end_time: number };
type TempEntry = {
  location: string;
  celsius: number;
  at: number;
  out_of_range?: boolean;
};
type NoteEntry = { text: string; at: number };

type TurnEvent =
  | { type: 'stack'; at: number; stt: string; llm: string; tts: string }
  | { type: 'user_text'; at: number; text: string }
  | { type: 'assistant_text'; at: number; text: string }
  | { type: 'tool_call'; at: number; name: string; args: unknown }
  | { type: 'stt_metrics'; at: number; duration: number; audio_duration: number }
  | {
      type: 'stt_compare';
      at: number;
      results: { label: string; text: string; ms: number }[];
    }
  | { type: 'eou_metrics'; at: number; transcription_delay: number }
  | {
      type: 'llm_metrics';
      at: number;
      duration: number;
      ttft: number;
      tokens: number;
      tok_per_sec: number;
    }
  | { type: 'tts_metrics'; at: number; duration: number; ttfb: number; chars: number }
  | { type: 'wake_gate'; at: number; forwarded: boolean; spot: string };

type StateEvent = {
  type: 'state';
  timers: ActiveTimer[];
  temperatures: TempEntry[];
  notes: NoteEntry[];
};

type AikaEvent = TurnEvent | StateEvent;

interface Turn {
  index: number;
  startedAt: number;
  userText?: string;
  assistantText?: string;
  toolCalls: { name: string; args: unknown }[];
  compare?: { label: string; text: string; ms: number }[];
  sttSec?: number;
  eouSec?: number;
  llmSec?: number;
  ttftSec?: number;
  tokens?: number;
  tokPerSec?: number;
  ttsSec?: number;
  ttfbSec?: number;
  chars?: number;
}

function s(n: number | undefined, digits = 2) {
  return n === undefined ? '—' : `${n.toFixed(digits)}s`;
}

/** Group a flat event log into per-turn objects. A new turn starts on each
 *  user_text event; later events go into the most recent turn. */
function groupTurns(events: TurnEvent[]): Turn[] {
  const turns: Turn[] = [];
  let current: Turn | null = null;
  // stt_compare is published just before the user_text it belongs to, so buffer
  // it and attach to the turn that the next user_text opens.
  let pendingCompare: Turn['compare'];
  for (const ev of events) {
    if (ev.type === 'stt_compare') {
      pendingCompare = ev.results;
      continue;
    }
    if (ev.type === 'user_text') {
      current = {
        index: turns.length + 1,
        startedAt: ev.at,
        userText: ev.text,
        toolCalls: [],
        compare: pendingCompare,
      };
      pendingCompare = undefined;
      turns.push(current);
      continue;
    }
    if (!current) {
      current = { index: 1, startedAt: ev.at, toolCalls: [] };
      turns.push(current);
    }
    switch (ev.type) {
      case 'assistant_text':
        current.assistantText = (current.assistantText ?? '') + ev.text;
        break;
      case 'tool_call':
        current.toolCalls.push({ name: ev.name, args: ev.args });
        break;
      case 'stt_metrics':
        current.sttSec = ev.duration;
        break;
      case 'eou_metrics':
        current.eouSec = ev.transcription_delay;
        break;
      case 'llm_metrics':
        current.llmSec = ev.duration;
        current.ttftSec = ev.ttft;
        current.tokens = ev.tokens;
        current.tokPerSec = ev.tok_per_sec;
        break;
      case 'tts_metrics':
        current.ttsSec = ev.duration;
        current.ttfbSec = ev.ttfb;
        current.chars = ev.chars;
        break;
    }
  }
  return turns;
}

function turnTotal(t: Turn): number | undefined {
  if (t.sttSec === undefined && t.llmSec === undefined && t.ttsSec === undefined) return undefined;
  return (t.sttSec ?? 0) + (t.llmSec ?? 0) + (t.ttsSec ?? 0);
}

function TurnCard({ turn }: { turn: Turn }) {
  const total = turnTotal(turn);
  const stt = turn.sttSec ?? 0;
  const llm = turn.llmSec ?? 0;
  const tts = turn.ttsSec ?? 0;
  const sum = stt + llm + tts || 1;
  return (
    <div className="border-border/60 border-t pt-2 first:border-t-0 first:pt-0">
      <div className="text-muted-foreground mb-1 flex justify-between text-[10px] tracking-wider uppercase">
        <span>turn {turn.index}</span>
        <span>total {s(total, 1)}</span>
      </div>
      {turn.userText !== undefined && (
        <div className="mb-0.5">
          <span className="text-muted-foreground">user:</span> {turn.userText}
        </div>
      )}
      {turn.compare && (
        <div className="bg-muted/40 mb-1 rounded p-1">
          <div className="text-muted-foreground mb-0.5 text-[9px] tracking-wider uppercase">
            stt compare
          </div>
          {turn.compare.map((r, i) => (
            <div key={i} className="flex gap-1">
              <span className="text-muted-foreground w-16 shrink-0">{r.label}</span>
              <span className="flex-1 break-words">{r.text}</span>
              <span className="text-muted-foreground shrink-0">{r.ms}ms</span>
            </div>
          ))}
        </div>
      )}
      {turn.toolCalls.map((tc, i) => (
        <div key={i} className="text-foreground/80 mb-0.5">
          <span className="text-muted-foreground">tool:</span> {tc.name}(
          {typeof tc.args === 'object' && tc.args !== null
            ? Object.entries(tc.args as Record<string, unknown>)
                .map(([k, v]) => `${k}=${JSON.stringify(v)}`)
                .join(', ')
            : String(tc.args)}
          )
        </div>
      ))}
      {turn.assistantText !== undefined && (
        <div className="mb-1">
          <span className="text-muted-foreground">aika:</span> {turn.assistantText}
        </div>
      )}
      {/* Per-stage breakdown + proportional bar */}
      <div className="text-muted-foreground mt-1 text-[10px]">
        STT {s(turn.sttSec)} · LLM {s(turn.llmSec)}
        {turn.ttftSec !== undefined && ` (ttft ${s(turn.ttftSec)})`}
        {turn.tokPerSec !== undefined && ` · ${turn.tokPerSec.toFixed(1)} t/s`} · TTS{' '}
        {s(turn.ttsSec)}
      </div>
      {total !== undefined && (
        <div className="mt-1 flex h-1 overflow-hidden rounded-sm">
          <div className="bg-blue-500" style={{ width: `${(stt / sum) * 100}%` }} />
          <div className="bg-green-500" style={{ width: `${(llm / sum) * 100}%` }} />
          <div className="bg-orange-500" style={{ width: `${(tts / sum) * 100}%` }} />
        </div>
      )}
    </div>
  );
}

function fmtRemaining(seconds: number): string {
  if (seconds <= 0) return 'done';
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return m > 0 ? `${m}m ${s}s` : `${s}s`;
}

export function DebugOverlay() {
  const [events, setEvents] = useState<TurnEvent[]>([]);

  useDataChannel('aika-debug', (msg) => {
    try {
      const text = new TextDecoder().decode(msg.payload);
      const parsed = JSON.parse(text) as AikaEvent;
      // state events are owned by LiveStatePanel — debug overlay only cares
      // about per-turn metrics and transcripts.
      if (parsed.type === 'state') return;
      setEvents((prev) => [...prev.slice(-99), parsed as TurnEvent]);
    } catch {
      // ignore
    }
  });

  // Stack info comes from the most recent stack event.
  const stack = useMemo(() => {
    for (let i = events.length - 1; i >= 0; i--) {
      if (events[i].type === 'stack') return events[i] as Extract<AikaEvent, { type: 'stack' }>;
    }
    return null;
  }, [events]);

  // Wake gate decisions get their own feed — blocked utterances never open a
  // turn, so they'd otherwise be invisible here.
  const gateEvents = useMemo(
    () =>
      events
        .filter((e): e is Extract<TurnEvent, { type: 'wake_gate' }> => e.type === 'wake_gate')
        .slice(-6)
        .reverse(),
    [events]
  );

  // Only show the last 3 turns to keep the overlay readable.
  const turns = useMemo(
    () =>
      groupTurns(events.filter((e) => e.type !== 'wake_gate'))
        .slice(-3)
        .reverse(),
    [events]
  );

  return (
    <div className="bg-background/95 fixed right-4 bottom-4 z-50 max-h-[60vh] w-80 overflow-y-auto rounded-lg border p-3 font-mono text-[11px] leading-snug shadow-lg backdrop-blur">
      <div className="text-muted-foreground mb-2 flex items-center justify-between text-[10px] tracking-wider uppercase">
        <span>aika debug</span>
        {stack && (
          <span>
            stt={stack.stt} · llm={stack.llm} · tts={stack.tts}
          </span>
        )}
      </div>
      {gateEvents.length > 0 && (
        <div className="bg-muted/40 mb-2 rounded p-1.5">
          <div className="text-muted-foreground mb-1 text-[9px] tracking-wider uppercase">
            wake gate (newest first)
          </div>
          {gateEvents.map((g, i) => (
            <div key={i} className="mb-0.5 flex gap-1.5">
              <span className={`w-14 shrink-0 ${g.forwarded ? 'text-green-500' : 'text-red-400'}`}>
                {g.forwarded ? 'sent' : 'blocked'}
              </span>
              <span className="text-foreground/80 flex-1 break-words">
                {g.spot ? `"${g.spot}"` : 'follow-up window was open'}
              </span>
            </div>
          ))}
        </div>
      )}
      {turns.length === 0 ? (
        <div className="text-muted-foreground mt-2 text-center text-[10px]">
          waiting for first turn...
        </div>
      ) : (
        <div className="mt-2 flex flex-col gap-2">
          {turns.map((t) => (
            <TurnCard key={t.index} turn={t} />
          ))}
        </div>
      )}
      <div className="text-muted-foreground mt-2 flex gap-3 border-t pt-1.5 text-[9px]">
        <span>
          <span className="inline-block size-2 rounded-sm bg-blue-500 align-middle" /> STT
        </span>
        <span>
          <span className="inline-block size-2 rounded-sm bg-green-500 align-middle" /> LLM
        </span>
        <span>
          <span className="inline-block size-2 rounded-sm bg-orange-500 align-middle" /> TTS
        </span>
      </div>
    </div>
  );
}
