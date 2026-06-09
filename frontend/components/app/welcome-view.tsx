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
}

interface StackPickerRowProps {
  label: string;
  value: StackChoice;
  cloudLabel: string;
  localLabel: string;
  onChange: (v: StackChoice) => void;
}

const StackPickerRow = ({
  label,
  value,
  cloudLabel,
  localLabel,
  onChange,
}: StackPickerRowProps) => {
  const baseBtn =
    'flex-1 rounded-md px-3 py-1.5 text-xs font-medium transition-colors';
  const active = 'bg-foreground text-background';
  const inactive = 'bg-muted text-muted-foreground hover:bg-muted/70';
  return (
    <div className="flex items-center gap-3">
      <span className="text-muted-foreground w-12 text-left text-xs font-mono uppercase tracking-wider">
        {label}
      </span>
      <div className="flex flex-1 gap-1 rounded-md bg-muted p-1">
        <button
          type="button"
          onClick={() => onChange('cloud')}
          className={`${baseBtn} ${value === 'cloud' ? active : inactive}`}
        >
          {cloudLabel}
        </button>
        <button
          type="button"
          onClick={() => onChange('local')}
          className={`${baseBtn} ${value === 'local' ? active : inactive}`}
        >
          {localLabel}
        </button>
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
  ref,
}: React.ComponentProps<'div'> & WelcomeViewProps) => {
  return (
    <div ref={ref}>
      <section className="bg-background flex flex-col items-center justify-center text-center">
        <WelcomeImage />

        <h1 className="text-foreground text-2xl font-bold tracking-tight">AIKA</h1>

        <div className="mt-6 flex w-80 flex-col gap-3">
          <StackPickerRow
            label="STT"
            value={stack.stt}
            cloudLabel="Deepgram nova-3"
            localLabel="Whisper medium"
            onChange={(v) => setStack({ ...stack, stt: v })}
          />
          <StackPickerRow
            label="LLM"
            value={stack.llm}
            cloudLabel="Groq Llama-3.3 70B"
            localLabel="Qwen2.5 7B"
            onChange={(v) => setStack({ ...stack, llm: v })}
          />
          <StackPickerRow
            label="TTS"
            value={stack.tts}
            cloudLabel="Cartesia"
            localLabel="Kokoro 82M"
            onChange={(v) => setStack({ ...stack, tts: v })}
          />
          <p className="text-muted-foreground mt-1 text-[10px] leading-snug text-center">
            Local stack runs on a CPU VPS — expect ~7-12s per turn.
          </p>
          <div className="mt-2">
            <div className="text-muted-foreground mb-1 text-[10px] uppercase tracking-wider">
              When should AIKA reply?
            </div>
            <div className="flex gap-1 rounded-md bg-muted p-1">
              {(
                [
                  {
                    mode: 'off' as WakeMode,
                    label: 'Always',
                    desc: 'AIKA replies to every utterance, even side conversations.',
                  },
                  {
                    mode: 'window' as WakeMode,
                    label: 'After "AIKA"',
                    desc:
                      'Say "AIKA" once. AIKA stays awake and listens for 30 seconds after each reply.',
                  },
                  {
                    mode: 'strict' as WakeMode,
                    label: 'Each command',
                    desc:
                      'Every command must start with "AIKA". Otherwise AIKA stays silent.',
                  },
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
            <p className="text-muted-foreground mt-1.5 text-[10px] leading-snug text-center">
              {wake === 'off'
                ? 'AIKA replies to every utterance, even side conversations.'
                : wake === 'window'
                  ? 'Say "AIKA" once. AIKA stays awake for 30 seconds after each reply.'
                  : 'Every command must start with "AIKA". Otherwise AIKA stays silent.'}
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
