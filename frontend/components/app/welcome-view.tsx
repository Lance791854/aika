import type { Stack, StackChoice, WakeMode } from '@/components/app/app';
import { Button } from '@/components/ui/button';

function WelcomeImage() {
  return (
    <svg
      width="64"
      height="64"
      viewBox="0 0 64 64"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      className="text-fg0 mb-4 size-16"
    >
      <path
        d="M15 24V40C15 40.7957 14.6839 41.5587 14.1213 42.1213C13.5587 42.6839 12.7956 43 12 43C11.2044 43 10.4413 42.6839 9.87868 42.1213C9.31607 41.5587 9 40.7957 9 40V24C9 23.2044 9.31607 22.4413 9.87868 21.8787C10.4413 21.3161 11.2044 21 12 21C12.7956 21 13.5587 21.3161 14.1213 21.8787C14.6839 22.4413 15 23.2044 15 24ZM22 5C21.2044 5 20.4413 5.31607 19.8787 5.87868C19.3161 6.44129 19 7.20435 19 8V56C19 56.7957 19.3161 57.5587 19.8787 58.1213C20.4413 58.6839 21.2044 59 22 59C22.7956 59 23.5587 58.6839 24.1213 58.1213C24.6839 57.5587 25 56.7957 25 56V8C25 7.20435 24.6839 6.44129 24.1213 5.87868C23.5587 5.31607 22.7956 5 22 5ZM32 13C31.2044 13 30.4413 13.3161 29.8787 13.8787C29.3161 14.4413 29 15.2044 29 16V48C29 48.7957 29.3161 49.5587 29.8787 50.1213C30.4413 50.6839 31.2044 51 32 51C32.7956 51 33.5587 50.6839 34.1213 50.1213C34.6839 49.5587 35 48.7957 35 48V16C35 15.2044 34.6839 14.4413 34.1213 13.8787C33.5587 13.3161 32.7956 13 32 13ZM42 21C41.2043 21 40.4413 21.3161 39.8787 21.8787C39.3161 22.4413 39 23.2044 39 24V40C39 40.7957 39.3161 41.5587 39.8787 42.1213C40.4413 42.6839 41.2043 43 42 43C42.7957 43 43.5587 42.6839 44.1213 42.1213C44.6839 41.5587 45 40.7957 45 40V24C45 23.2044 44.6839 22.4413 44.1213 21.8787C43.5587 21.3161 42.7957 21 42 21ZM52 17C51.2043 17 50.4413 17.3161 49.8787 17.8787C49.3161 18.4413 49 19.2044 49 20V44C49 44.7957 49.3161 45.5587 49.8787 46.1213C50.4413 46.6839 51.2043 47 52 47C52.7957 47 53.5587 46.6839 54.1213 46.1213C54.6839 45.5587 55 44.7957 55 44V20C55 19.2044 54.6839 18.4413 54.1213 17.8787C53.5587 17.3161 52.7957 17 52 17Z"
        fill="currentColor"
      />
    </svg>
  );
}

interface WelcomeViewProps {
  startButtonText: string;
  onStartCall: () => void;
  stack: Stack;
  setStack: (s: Stack) => void;
  debug: boolean;
  setDebug: (v: boolean) => void;
  wake: WakeMode;
  setWake: (v: WakeMode) => void;
  compare: boolean;
  setCompare: (v: boolean) => void;
  chef: string;
  setChef: (v: string) => void;
}

// Where each option actually runs: hosted API, Cloudflare, our CPU VPS, or the RunPod GPU.
type Tier = 'API' | 'CF API' | 'CPU' | 'GPU';

const TIER_CLASS: Record<Tier, { active: string; inactive: string }> = {
  API: { active: 'text-sky-600', inactive: 'text-sky-500/70' },
  'CF API': { active: 'text-violet-600', inactive: 'text-violet-500/70' },
  CPU: { active: 'text-amber-600', inactive: 'text-amber-500/70' },
  GPU: { active: 'text-emerald-600', inactive: 'text-emerald-500/70' },
};

interface StackOption {
  value: StackChoice;
  label: string;
  tier: Tier;
}

interface StackPickerRowProps {
  label: string;
  value: StackChoice;
  options: StackOption[];
  onChange: (v: StackChoice) => void;
}

const StackPickerRow = ({ label, value, options, onChange }: StackPickerRowProps) => {
  const baseBtn =
    'flex min-h-11 flex-1 flex-col items-center justify-center rounded-md px-1.5 py-1 text-xs font-medium transition-colors';
  const active = 'bg-foreground text-background';
  const inactive = 'bg-muted text-muted-foreground hover:bg-muted/70';
  return (
    <div className="flex items-center gap-3">
      <span className="text-muted-foreground w-12 text-left font-mono text-xs tracking-wider uppercase">
        {label}
      </span>
      <div className="bg-muted flex flex-1 gap-1 rounded-md p-1">
        {options.map((o) => {
          const isActive = value === o.value;
          return (
            <button
              key={o.value}
              type="button"
              onClick={() => onChange(o.value)}
              className={`${baseBtn} ${isActive ? active : inactive}`}
            >
              <span className="block leading-tight">{o.label}</span>
              <span
                className={`block text-[9px] font-semibold tracking-widest ${
                  TIER_CLASS[o.tier][isActive ? 'active' : 'inactive']
                }`}
              >
                {o.tier}
              </span>
            </button>
          );
        })}
      </div>
    </div>
  );
};

export const WelcomeView = ({
  startButtonText,
  onStartCall,
  stack,
  setStack,
  debug,
  setDebug,
  wake,
  setWake,
  compare,
  setCompare,
  chef,
  setChef,
  ref,
}: React.ComponentProps<'div'> & WelcomeViewProps) => {
  return (
    <div ref={ref}>
      <section className="bg-background flex flex-col items-center justify-center text-center">
        <WelcomeImage />

        <h1 className="text-foreground text-2xl font-bold tracking-tight">AIKA</h1>

        <div className="mt-6 flex w-[26rem] max-w-[calc(100vw-2rem)] flex-col gap-3">
          <div className="flex items-center gap-3">
            <span className="text-muted-foreground w-12 text-left font-mono text-xs tracking-wider uppercase">
              Chef
            </span>
            <input
              type="text"
              value={chef}
              onChange={(e) => setChef(e.target.value)}
              placeholder="chef's name"
              maxLength={24}
              className="bg-muted text-foreground placeholder:text-muted-foreground flex-1 rounded-md px-3 py-1.5 text-xs outline-none"
            />
          </div>
          <StackPickerRow
            label="STT"
            value={stack.stt}
            options={[
              { value: 'cloud', label: 'Deepgram', tier: 'API' },
              { value: 'cartesia', label: 'Cartesia', tier: 'API' },
              { value: 'local', label: 'Whisper', tier: 'CPU' },
              { value: 'gpu', label: 'Parakeet', tier: 'GPU' },
              { value: 'cf', label: 'Nova-3 CF', tier: 'CF API' },
            ]}
            onChange={(v) => setStack({ ...stack, stt: v })}
          />
          <StackPickerRow
            label="LLM"
            value={stack.llm}
            options={[
              { value: 'cloud', label: 'Llama 70B Groq', tier: 'API' },
              { value: 'local', label: 'Qwen2.5 7B', tier: 'CPU' },
              { value: 'gpu', label: 'Qwen3 8B', tier: 'GPU' },
              { value: 'cf', label: 'Llama 70B CF', tier: 'CF API' },
            ]}
            onChange={(v) => setStack({ ...stack, llm: v })}
          />
          <StackPickerRow
            label="TTS"
            value={stack.tts}
            options={[
              { value: 'cloud', label: 'Cartesia', tier: 'API' },
              { value: 'local', label: 'Kokoro 82M', tier: 'CPU' },
              { value: 'gpu', label: 'Kokoro 82M', tier: 'GPU' },
              { value: 'cf', label: 'Aura 2 CF', tier: 'CF API' },
            ]}
            onChange={(v) => setStack({ ...stack, tts: v })}
          />
          <p className="text-muted-foreground mt-1 text-center text-[10px] leading-snug">
            <span className="font-semibold text-sky-500/80">API</span> = hosted cloud &middot;{' '}
            <span className="font-semibold text-violet-500/80">CF API</span> = Cloudflare &middot;{' '}
            <span className="font-semibold text-amber-500/80">CPU</span> = self-hosted VPS
            (~7-12s/turn) &middot; <span className="font-semibold text-emerald-500/80">GPU</span> =
            self-hosted RunPod
          </p>
          <div className="mt-2">
            <div className="text-muted-foreground mb-1 text-[10px] tracking-wider uppercase">
              When should AIKA reply?
            </div>
            <div className="bg-muted flex gap-1 rounded-md p-1">
              {(
                [
                  { mode: 'off' as WakeMode, label: 'Always' },
                  { mode: 'strict' as WakeMode, label: 'After "AIKA"' },
                  { mode: 'device' as WakeMode, label: 'Wearable sim' },
                ] as const
              ).map(({ mode, label }) => {
                const active = wake === mode;
                return (
                  <button
                    key={mode}
                    type="button"
                    onClick={() => setWake(mode)}
                    className={`flex-1 rounded-md px-2 py-1 text-xs font-medium transition-colors ${
                      active
                        ? 'bg-foreground text-background'
                        : 'text-muted-foreground hover:bg-muted/70'
                    }`}
                  >
                    {label}
                  </button>
                );
              })}
            </div>
            <p className="text-muted-foreground mt-1.5 text-center text-[10px] leading-snug">
              {wake === 'off' && 'AIKA replies to everything it hears, even side conversations.'}
              {wake === 'strict' &&
                'Only replies when you address it by name, like "AIKA ..." or "hey AIKA ...". Otherwise it stays silent.'}
              {wake === 'device' &&
                'Works like the planned wearable. A small model on our own server listens for "AIKA". Nothing is sent to the cloud until it hears that.'}
            </p>
          </div>
          <button
            type="button"
            onClick={() => setDebug(!debug)}
            className={`rounded-md py-1.5 text-xs font-medium transition-colors ${
              debug
                ? 'bg-foreground text-background'
                : 'bg-muted text-muted-foreground hover:bg-muted/70'
            }`}
          >
            Debug overlay: {debug ? 'on' : 'off'}
          </button>
          {debug && (
            <button
              type="button"
              onClick={() => setCompare(!compare)}
              className={`rounded-md py-1.5 text-xs font-medium transition-colors ${
                compare
                  ? 'bg-foreground text-background'
                  : 'bg-muted text-muted-foreground hover:bg-muted/70'
              }`}
            >
              Compare STT (Parakeet vs Deepgram): {compare ? 'on' : 'off'}
            </button>
          )}
        </div>

        <Button
          size="lg"
          onClick={onStartCall}
          className="mt-6 w-64 rounded-full font-mono text-xs font-bold tracking-wider uppercase"
        >
          {startButtonText}
        </Button>
      </section>
    </div>
  );
};
