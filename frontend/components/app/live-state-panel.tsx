'use client';

import { useEffect, useState } from 'react';
import { useDataChannel } from '@livekit/components-react';

type ActiveTimer = { name: string; end_time: number };
type TempEntry = {
  location: string;
  celsius: number;
  at: number;
  out_of_range?: boolean;
};
type NoteEntry = { text: string; at: number };

type StateEvent = {
  type: 'state';
  timers: ActiveTimer[];
  temperatures: TempEntry[];
  notes: NoteEntry[];
};

function fmtRemaining(seconds: number): string {
  if (seconds <= 0) return 'done';
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  if (m >= 60) {
    const h = Math.floor(m / 60);
    return `${h}h ${m % 60}m`;
  }
  return m > 0 ? `${m}m ${s.toString().padStart(2, '0')}s` : `${s}s`;
}

function fmtTimeAgo(at: number, now: number): string {
  const seconds = Math.max(0, Math.floor((now / 1000 - at)));
  if (seconds < 60) return `${seconds}s ago`;
  const m = Math.floor(seconds / 60);
  if (m < 60) return `${m}m ago`;
  const h = Math.floor(m / 60);
  return `${h}h ago`;
}

function TimerCard({ timer, now }: { timer: ActiveTimer; now: number }) {
  const remaining = timer.end_time - now / 1000;
  // not perfectly accurate without start_time, but good enough as a visual.
  const total = Math.max(remaining + 60, 60);
  const progress = Math.max(0, Math.min(1, remaining / total));
  const danger = remaining < 30 && remaining > 0;
  return (
    <div className="bg-card border-border/60 rounded-lg border p-3">
      <div className="flex items-baseline justify-between">
        <div className="text-foreground truncate text-sm font-medium capitalize">
          {timer.name}
        </div>
        <div
          className={`font-mono text-lg tabular-nums ${
            danger ? 'text-red-500' : 'text-foreground'
          }`}
        >
          {fmtRemaining(remaining)}
        </div>
      </div>
      <div className="bg-muted mt-2 h-1.5 overflow-hidden rounded-full">
        <div
          className={`h-full transition-all duration-1000 ${
            danger ? 'bg-red-500' : 'bg-foreground'
          }`}
          style={{ width: `${progress * 100}%` }}
        />
      </div>
    </div>
  );
}

function TempCard({
  temp,
  now,
  onInject,
}: {
  temp: TempEntry;
  now: number;
  onInject: (location: string, severity: 'low' | 'ok' | 'high') => void;
}) {
  const c = temp.celsius;
  const display = Number.isInteger(c) ? c : c.toFixed(1);
  return (
    <div
      className={`rounded-md border p-2 ${
        temp.out_of_range
          ? 'border-red-500/50 bg-red-500/5'
          : 'border-border/60 bg-card'
      }`}
    >
      <div className="flex items-center justify-between">
        <div className="min-w-0">
          <div className="text-foreground truncate text-sm font-medium capitalize">
            {temp.location}
          </div>
          <div className="text-muted-foreground text-[10px]">
            {fmtTimeAgo(temp.at, now)}
          </div>
        </div>
        <div
          className={`font-mono text-base tabular-nums ${
            temp.out_of_range ? 'text-red-500 font-bold' : 'text-foreground'
          }`}
        >
          {display}°C
        </div>
      </div>
      {/* debug: inject a fake reading at the chosen severity to test alerts */}
      <div className="mt-1.5 flex gap-1">
        <button
          type="button"
          onClick={() => onInject(temp.location, 'low')}
          className="text-muted-foreground hover:text-foreground bg-muted/60 hover:bg-muted flex-1 rounded px-1.5 py-0.5 text-[10px]"
          title={`Log ${temp.location} below safe range`}
        >
          low
        </button>
        <button
          type="button"
          onClick={() => onInject(temp.location, 'ok')}
          className="text-muted-foreground hover:text-foreground bg-muted/60 hover:bg-muted flex-1 rounded px-1.5 py-0.5 text-[10px]"
          title={`Log ${temp.location} in safe range`}
        >
          ok
        </button>
        <button
          type="button"
          onClick={() => onInject(temp.location, 'high')}
          className="text-muted-foreground hover:text-foreground bg-muted/60 hover:bg-muted flex-1 rounded px-1.5 py-0.5 text-[10px]"
          title={`Log ${temp.location} above safe range`}
        >
          high
        </button>
      </div>
    </div>
  );
}

function NoteCard({ note, now }: { note: NoteEntry; now: number }) {
  return (
    <div className="bg-card border-border/60 rounded-md border p-2">
      <div className="text-foreground text-sm leading-snug">{note.text}</div>
      <div className="text-muted-foreground mt-1 text-[10px]">
        {fmtTimeAgo(note.at, now)}
      </div>
    </div>
  );
}

interface Section {
  key: 'timers' | 'temperatures' | 'notes';
  title: string;
  count: number;
  empty: string;
}

export function LiveStatePanel() {
  const [state, setState] = useState<{
    timers: ActiveTimer[];
    temperatures: TempEntry[];
    notes: NoteEntry[];
  }>({ timers: [], temperatures: [], notes: [] });
  const [collapsed, setCollapsed] = useState(false);

  // hidden-section toggles, persisted across tab in localStorage
  const [hidden, setHidden] = useState<Set<string>>(() => {
    if (typeof window === 'undefined') return new Set();
    try {
      return new Set(JSON.parse(window.localStorage.getItem('aika-panel-hidden') ?? '[]'));
    } catch {
      return new Set();
    }
  });
  useEffect(() => {
    if (typeof window !== 'undefined') {
      window.localStorage.setItem('aika-panel-hidden', JSON.stringify([...hidden]));
    }
  }, [hidden]);

  // tick each second so timer countdowns update
  const [now, setNow] = useState<number>(() => Date.now());
  useEffect(() => {
    const id = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(id);
  }, []);

  const { send } = useDataChannel('aika-debug', (msg) => {
    try {
      const text = new TextDecoder().decode(msg.payload);
      const parsed = JSON.parse(text);
      if (parsed.type === 'state') {
        const ev = parsed as StateEvent;
        setState({
          timers: ev.timers ?? [],
          temperatures: ev.temperatures ?? [],
          notes: ev.notes ?? [],
        });
      }
    } catch {
      // ignore
    }
  });

  const runSafetyCheck = () => {
    const payload = new TextEncoder().encode(JSON.stringify({ type: 'check_temps' }));
    send(payload, { reliable: true });
  };

  const injectTemp = (location: string, severity: 'low' | 'ok' | 'high') => {
    const payload = new TextEncoder().encode(
      JSON.stringify({ type: 'inject_temp', location, severity }),
    );
    send(payload, { reliable: true });
  };

  const sections: Section[] = [
    {
      key: 'timers',
      title: 'Timers',
      count: state.timers.length,
      empty: 'No active timers.',
    },
    {
      key: 'temperatures',
      title: 'Temperatures',
      count: state.temperatures.length,
      empty: 'No readings yet.',
    },
    {
      key: 'notes',
      title: 'Notes',
      count: state.notes.length,
      empty: 'No notes yet.',
    },
  ];

  const toggleHidden = (key: string) => {
    setHidden((prev) => {
      const next = new Set(prev);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });
  };

  if (collapsed) {
    return (
      <button
        type="button"
        onClick={() => setCollapsed(false)}
        className="bg-background/90 hover:bg-background border-border fixed top-4 right-4 z-40 rounded-lg border p-2 font-mono text-xs shadow-md backdrop-blur"
        title="Show AIKA status"
      >
        Status
      </button>
    );
  }

  return (
    <aside className="bg-background/95 border-border fixed top-4 right-4 z-40 flex max-h-[calc(100vh-2rem)] w-72 flex-col overflow-hidden rounded-lg border shadow-md backdrop-blur">
      <header className="border-border/60 flex items-center justify-between border-b px-3 py-2">
        <h2 className="text-foreground text-xs font-semibold uppercase tracking-wider">
          AIKA Status
        </h2>
        <button
          type="button"
          onClick={() => setCollapsed(true)}
          className="text-muted-foreground hover:text-foreground text-xs"
          title="Hide"
        >
          ×
        </button>
      </header>
      <div className="flex gap-1 border-b px-2 py-1.5">
        {sections.map((s) => (
          <button
            key={s.key}
            type="button"
            onClick={() => toggleHidden(s.key)}
            className={`flex-1 rounded-md px-2 py-1 text-[10px] font-medium uppercase tracking-wider transition-colors ${
              hidden.has(s.key)
                ? 'text-muted-foreground bg-muted/40'
                : 'text-foreground bg-muted'
            }`}
            title={hidden.has(s.key) ? `Show ${s.title}` : `Hide ${s.title}`}
          >
            {s.title} {s.count > 0 && `(${s.count})`}
          </button>
        ))}
      </div>
      <div className="overflow-y-auto p-3">
        {sections.filter((s) => !hidden.has(s.key)).length === 0 && (
          <div className="text-muted-foreground py-4 text-center text-xs">
            All sections hidden.
          </div>
        )}

        {!hidden.has('timers') && (
          <section className="mb-4">
            <h3 className="text-muted-foreground mb-2 text-[10px] font-semibold uppercase tracking-wider">
              Timers
            </h3>
            {state.timers.length === 0 ? (
              <p className="text-muted-foreground text-xs">No active timers.</p>
            ) : (
              <div className="flex flex-col gap-2">
                {state.timers.map((t) => (
                  <TimerCard key={t.name} timer={t} now={now} />
                ))}
              </div>
            )}
          </section>
        )}

        {!hidden.has('temperatures') && (
          <section className="mb-4">
            <div className="mb-2 flex items-center justify-between">
              <h3 className="text-muted-foreground text-[10px] font-semibold uppercase tracking-wider">
                Temperatures
              </h3>
              <button
                type="button"
                onClick={runSafetyCheck}
                className="text-foreground bg-muted hover:bg-muted/70 rounded-md px-2 py-0.5 text-[10px] font-medium uppercase tracking-wider"
                title="AIKA scans recent readings and speaks any out-of-range warnings"
              >
                Run check
              </button>
            </div>
            {state.temperatures.length === 0 ? (
              <p className="text-muted-foreground text-xs">No readings yet.</p>
            ) : (
              <div className="flex flex-col gap-1.5">
                {state.temperatures.map((t, i) => (
                  <TempCard key={i} temp={t} now={now} onInject={injectTemp} />
                ))}
              </div>
            )}
          </section>
        )}

        {!hidden.has('notes') && (
          <section>
            <h3 className="text-muted-foreground mb-2 text-[10px] font-semibold uppercase tracking-wider">
              Notes
            </h3>
            {state.notes.length === 0 ? (
              <p className="text-muted-foreground text-xs">No notes yet.</p>
            ) : (
              <div className="flex flex-col gap-1.5">
                {state.notes.map((n, i) => (
                  <NoteCard key={i} note={n} now={now} />
                ))}
              </div>
            )}
          </section>
        )}
      </div>
    </aside>
  );
}
