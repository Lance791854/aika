'use client';

import { useEffect, useRef, useState } from 'react';
import { useDataChannel } from '@livekit/components-react';

type ActiveTimer = { name: string; end_time: number };
type ReminderEntry = { text: string; end_time: number };
type TempEntry = {
  location: string;
  celsius: number;
  at: number;
  out_of_range?: boolean;
};
type NoteEntry = { text: string; at: number };
type UnhandledEntry = { text: string; at: number };
type StockEntry = { item: string; quantity: string; urgency: string; at: number };

type StateEvent = {
  type: 'state';
  timers: ActiveTimer[];
  reminders: ReminderEntry[];
  temperatures: TempEntry[];
  stock: StockEntry[];
  notes: NoteEntry[];
  unhandled: UnhandledEntry[];
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
  const seconds = Math.max(0, Math.floor(now / 1000 - at));
  if (seconds < 60) return `${seconds}s ago`;
  const m = Math.floor(seconds / 60);
  if (m < 60) return `${m}m ago`;
  const h = Math.floor(m / 60);
  return `${h}h ago`;
}

function DeleteX({ onClick, title }: { onClick: () => void; title: string }) {
  return (
    <button
      type="button"
      onClick={onClick}
      title={title}
      className="text-muted-foreground shrink-0 px-1 text-sm leading-none hover:text-red-500"
    >
      ×
    </button>
  );
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
        <div className="text-foreground truncate text-sm font-medium capitalize">{timer.name}</div>
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

function ReminderCard({ reminder, now }: { reminder: ReminderEntry; now: number }) {
  const remaining = reminder.end_time - now / 1000;
  const danger = remaining < 30 && remaining > 0;
  return (
    <div className="bg-card border-border/60 rounded-lg border p-3">
      <div className="flex items-baseline justify-between gap-2">
        <div className="text-foreground min-w-0 truncate text-sm">{reminder.text}</div>
        <div
          className={`shrink-0 font-mono text-sm tabular-nums ${
            danger ? 'text-red-500' : 'text-muted-foreground'
          }`}
        >
          in {fmtRemaining(remaining)}
        </div>
      </div>
    </div>
  );
}

function TempCard({
  temp,
  now,
  onInject,
  onDelete,
}: {
  temp: TempEntry;
  now: number;
  onInject: (location: string, severity: 'low' | 'ok' | 'high') => void;
  onDelete: () => void;
}) {
  const c = temp.celsius;
  const display = Number.isInteger(c) ? c : c.toFixed(1);
  return (
    <div
      className={`rounded-md border p-2 ${
        temp.out_of_range ? 'border-red-500/50 bg-red-500/5' : 'border-border/60 bg-card'
      }`}
    >
      <div className="flex items-center justify-between">
        <div className="min-w-0">
          <div className="text-foreground truncate text-sm font-medium capitalize">
            {temp.location}
          </div>
          <div className="text-muted-foreground text-[10px]">{fmtTimeAgo(temp.at, now)}</div>
        </div>
        <div className="flex items-center gap-1">
          <div
            className={`font-mono text-base tabular-nums ${
              temp.out_of_range ? 'font-bold text-red-500' : 'text-foreground'
            }`}
          >
            {display}°C
          </div>
          <DeleteX onClick={onDelete} title={`Remove ${temp.location}`} />
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

function StockCard({
  entry,
  now,
  onDelete,
}: {
  entry: StockEntry;
  now: number;
  onDelete: () => void;
}) {
  const urgent = entry.urgency === 'urgent';
  return (
    <div
      className={`rounded-md border p-2 ${
        urgent ? 'border-red-500/50 bg-red-500/5' : 'border-border/60 bg-card'
      }`}
    >
      <div className="flex items-start justify-between gap-1">
        <div className="text-foreground text-sm leading-snug">
          {entry.item}
          {entry.quantity && <span className="text-muted-foreground"> — {entry.quantity}</span>}
          {urgent && <span className="ml-1 font-semibold text-red-500">urgent</span>}
        </div>
        <DeleteX onClick={onDelete} title="Delete stock request" />
      </div>
      <div className="text-muted-foreground mt-1 text-[10px]">{fmtTimeAgo(entry.at, now)}</div>
    </div>
  );
}

function NoteCard({ note, now, onDelete }: { note: NoteEntry; now: number; onDelete: () => void }) {
  return (
    <div className="bg-card border-border/60 rounded-md border p-2">
      <div className="flex items-start justify-between gap-1">
        <div className="text-foreground text-sm leading-snug">{note.text}</div>
        <DeleteX onClick={onDelete} title="Delete note" />
      </div>
      <div className="text-muted-foreground mt-1 text-[10px]">{fmtTimeAgo(note.at, now)}</div>
    </div>
  );
}

function UnhandledCard({
  entry,
  now,
  onDelete,
}: {
  entry: UnhandledEntry;
  now: number;
  onDelete: () => void;
}) {
  return (
    <div className="rounded-md border border-amber-500/40 bg-amber-500/5 p-2">
      <div className="flex items-start justify-between gap-1">
        <div className="text-foreground text-sm leading-snug">{entry.text}</div>
        <DeleteX onClick={onDelete} title="Delete entry" />
      </div>
      <div className="text-muted-foreground mt-1 text-[10px]">{fmtTimeAgo(entry.at, now)}</div>
    </div>
  );
}

interface Section {
  key: 'timers' | 'reminders' | 'temperatures' | 'stock' | 'notes' | 'unhandled';
  title: string;
  count: number;
  empty: string;
}

export function LiveStatePanel() {
  const [state, setState] = useState<{
    timers: ActiveTimer[];
    reminders: ReminderEntry[];
    temperatures: TempEntry[];
    stock: StockEntry[];
    notes: NoteEntry[];
    unhandled: UnhandledEntry[];
  }>({ timers: [], reminders: [], temperatures: [], stock: [], notes: [], unhandled: [] });
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

  const gotState = useRef(false);

  const { send } = useDataChannel('aika-debug', (msg) => {
    try {
      const text = new TextDecoder().decode(msg.payload);
      const parsed = JSON.parse(text);
      if (parsed.type === 'state') {
        gotState.current = true;
        const ev = parsed as StateEvent;
        setState({
          timers: ev.timers ?? [],
          reminders: ev.reminders ?? [],
          temperatures: ev.temperatures ?? [],
          stock: ev.stock ?? [],
          notes: ev.notes ?? [],
          unhandled: ev.unhandled ?? [],
        });
      }
    } catch {
      // ignore
    }
  });

  // The agent dumps state once on join, which can arrive before this
  // component is listening. Ask for it until the first snapshot lands.
  useEffect(() => {
    let tries = 0;
    const id = setInterval(() => {
      if (gotState.current || tries++ >= 5) {
        clearInterval(id);
        return;
      }
      try {
        send(new TextEncoder().encode(JSON.stringify({ type: 'get_state' })), {
          reliable: true,
        });
      } catch {
        // data channel not up yet — try again next tick
      }
    }, 1500);
    return () => clearInterval(id);
  }, [send]);

  const runSafetyCheck = () => {
    const payload = new TextEncoder().encode(JSON.stringify({ type: 'check_temps' }));
    send(payload, { reliable: true });
  };

  const injectTemp = (location: string, severity: 'low' | 'ok' | 'high') => {
    const payload = new TextEncoder().encode(
      JSON.stringify({ type: 'inject_temp', location, severity })
    );
    send(payload, { reliable: true });
  };

  const act = (msg: Record<string, unknown>) => {
    send(new TextEncoder().encode(JSON.stringify(msg)), { reliable: true });
  };
  const clearUnhandled = () => act({ type: 'clear_unhandled' });
  const deleteUnhandled = (at: number) => act({ type: 'delete_unhandled', at });
  const clearNotes = () => act({ type: 'clear_notes' });
  const clearStock = () => act({ type: 'clear_stock' });
  const deleteStock = (at: number) => act({ type: 'delete_stock', at });
  const deleteNote = (at: number) => act({ type: 'delete_note', at });
  const clearTemps = () => act({ type: 'clear_temps' });
  const deleteTemp = (location: string) => act({ type: 'delete_temp', location });

  const sections: Section[] = [
    {
      key: 'timers',
      title: 'Timers',
      count: state.timers.length,
      empty: 'No active timers.',
    },
    {
      key: 'reminders',
      title: 'Reminders',
      count: state.reminders.length,
      empty: 'No reminders set.',
    },
    {
      key: 'temperatures',
      title: 'Temperatures',
      count: state.temperatures.length,
      empty: 'No readings yet.',
    },
    {
      key: 'stock',
      title: 'Stock',
      count: state.stock.length,
      empty: 'No stock requests.',
    },
    {
      key: 'notes',
      title: 'Notes',
      count: state.notes.length,
      empty: 'No notes yet.',
    },
    {
      key: 'unhandled',
      title: 'Unhandled',
      count: state.unhandled.length,
      empty: 'Nothing unhandled.',
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
        <h2 className="text-foreground text-xs font-semibold tracking-wider uppercase">
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
      <div className="grid grid-cols-2 gap-1 border-b px-2 py-1.5">
        {sections.map((s) => (
          <button
            key={s.key}
            type="button"
            onClick={() => toggleHidden(s.key)}
            className={`truncate rounded-md px-2 py-1 text-[10px] font-medium tracking-wider uppercase transition-colors ${
              hidden.has(s.key) ? 'text-muted-foreground bg-muted/40' : 'text-foreground bg-muted'
            }`}
            title={hidden.has(s.key) ? `Show ${s.title}` : `Hide ${s.title}`}
          >
            {s.title} {s.count > 0 && `(${s.count})`}
          </button>
        ))}
      </div>
      <div className="overflow-y-auto p-3">
        {sections.filter((s) => !hidden.has(s.key)).length === 0 && (
          <div className="text-muted-foreground py-4 text-center text-xs">All sections hidden.</div>
        )}

        {!hidden.has('timers') && (
          <section className="mb-4">
            <h3 className="text-muted-foreground mb-2 text-[10px] font-semibold tracking-wider uppercase">
              Timers
            </h3>
            {state.timers.length === 0 ? (
              <p className="text-muted-foreground text-xs">
                {'No active timers. Try "steak timer 4 minutes".'}
              </p>
            ) : (
              <div className="flex flex-col gap-2">
                {state.timers.map((t) => (
                  <TimerCard key={t.name} timer={t} now={now} />
                ))}
              </div>
            )}
          </section>
        )}

        {!hidden.has('reminders') && (
          <section className="mb-4">
            <h3 className="text-muted-foreground mb-2 text-[10px] font-semibold tracking-wider uppercase">
              Reminders
            </h3>
            {state.reminders.length === 0 ? (
              <p className="text-muted-foreground text-xs">
                {'No reminders set. Try "remind me in 20 minutes to rotate the stock".'}
              </p>
            ) : (
              <div className="flex flex-col gap-2">
                {state.reminders.map((r, i) => (
                  <ReminderCard key={i} reminder={r} now={now} />
                ))}
              </div>
            )}
          </section>
        )}

        {!hidden.has('temperatures') && (
          <section className="mb-4">
            <div className="mb-2 flex items-center justify-between">
              <h3 className="text-muted-foreground text-[10px] font-semibold tracking-wider uppercase">
                Temperatures
              </h3>
              <div className="flex gap-1">
                <button
                  type="button"
                  onClick={runSafetyCheck}
                  className="text-foreground bg-muted hover:bg-muted/70 rounded-md px-2 py-0.5 text-[10px] font-medium tracking-wider uppercase"
                  title="AIKA scans recent readings and speaks any out-of-range warnings"
                >
                  Run check
                </button>
                {state.temperatures.length > 0 && (
                  <button
                    type="button"
                    onClick={clearTemps}
                    className="text-foreground bg-muted hover:bg-muted/70 rounded-md px-2 py-0.5 text-[10px] font-medium tracking-wider uppercase"
                    title="Clear all temperature readings"
                  >
                    Clear
                  </button>
                )}
              </div>
            </div>
            {state.temperatures.length === 0 ? (
              <p className="text-muted-foreground text-xs">
                {'No readings yet. Try "log the freezer at minus 18".'}
              </p>
            ) : (
              <div className="flex flex-col gap-1.5">
                {state.temperatures.map((t, i) => (
                  <TempCard
                    key={i}
                    temp={t}
                    now={now}
                    onInject={injectTemp}
                    onDelete={() => deleteTemp(t.location)}
                  />
                ))}
              </div>
            )}
          </section>
        )}

        {!hidden.has('stock') && (
          <section className="mb-4">
            <div className="mb-2 flex items-center justify-between">
              <h3 className="text-muted-foreground text-[10px] font-semibold tracking-wider uppercase">
                Stock
              </h3>
              {state.stock.length > 0 && (
                <button
                  type="button"
                  onClick={clearStock}
                  className="text-foreground bg-muted hover:bg-muted/70 rounded-md px-2 py-0.5 text-[10px] font-medium tracking-wider uppercase"
                  title="Clear all stock requests"
                >
                  Clear
                </button>
              )}
            </div>
            {state.stock.length === 0 ? (
              <p className="text-muted-foreground text-xs">
                {'No stock requests. Try "we\'re low on cream, order five kilos".'}
              </p>
            ) : (
              <div className="flex flex-col gap-1.5">
                {state.stock.map((e, i) => (
                  <StockCard key={i} entry={e} now={now} onDelete={() => deleteStock(e.at)} />
                ))}
              </div>
            )}
          </section>
        )}

        {!hidden.has('notes') && (
          <section>
            <div className="mb-2 flex items-center justify-between">
              <h3 className="text-muted-foreground text-[10px] font-semibold tracking-wider uppercase">
                Notes
              </h3>
              {state.notes.length > 0 && (
                <button
                  type="button"
                  onClick={clearNotes}
                  className="text-foreground bg-muted hover:bg-muted/70 rounded-md px-2 py-0.5 text-[10px] font-medium tracking-wider uppercase"
                  title="Clear all notes"
                >
                  Clear
                </button>
              )}
            </div>
            {state.notes.length === 0 ? (
              <p className="text-muted-foreground text-xs">
                {'No notes yet. Try "make a note we\'re out of butter".'}
              </p>
            ) : (
              <div className="flex flex-col gap-1.5">
                {state.notes.map((n, i) => (
                  <NoteCard key={i} note={n} now={now} onDelete={() => deleteNote(n.at)} />
                ))}
              </div>
            )}
          </section>
        )}

        {!hidden.has('unhandled') && (
          <section className="mt-4">
            <div className="mb-2 flex items-center justify-between">
              <h3 className="text-muted-foreground text-[10px] font-semibold tracking-wider uppercase">
                Unhandled
              </h3>
              {state.unhandled.length > 0 && (
                <button
                  type="button"
                  onClick={clearUnhandled}
                  className="text-foreground bg-muted hover:bg-muted/70 rounded-md px-2 py-0.5 text-[10px] font-medium tracking-wider uppercase"
                  title="Clear all unhandled requests"
                >
                  Clear
                </button>
              )}
            </div>
            {state.unhandled.length === 0 ? (
              <p className="text-muted-foreground text-xs">
                Nothing unhandled. Requests AIKA can&apos;t do appear here.
              </p>
            ) : (
              <div className="flex flex-col gap-1.5">
                {state.unhandled.map((u, i) => (
                  <UnhandledCard
                    key={i}
                    entry={u}
                    now={now}
                    onDelete={() => deleteUnhandled(u.at)}
                  />
                ))}
              </div>
            )}
          </section>
        )}
      </div>
    </aside>
  );
}
