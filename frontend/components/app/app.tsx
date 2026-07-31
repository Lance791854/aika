'use client';

import { useMemo, useRef, useState } from 'react';
import { TokenSource } from 'livekit-client';
import { useSession } from '@livekit/components-react';
import { WarningIcon } from '@phosphor-icons/react/dist/ssr';
import type { AppConfig } from '@/app-config';
import { AgentSessionProvider } from '@/components/agents-ui/agent-session-provider';
import { StartAudioButton } from '@/components/agents-ui/start-audio-button';
import { ViewController } from '@/components/app/view-controller';
import { Toaster } from '@/components/ui/sonner';
import { useAgentErrors } from '@/hooks/useAgentErrors';
import { useDebugMode } from '@/hooks/useDebug';
import { getSandboxTokenSource } from '@/lib/utils';

export type StackChoice = 'cloud' | 'local' | 'gpu' | 'cartesia' | 'cf';
export type Stack = { stt: StackChoice; llm: StackChoice; tts: StackChoice };
export type WakeMode = 'off' | 'strict' | 'device';

const DEFAULT_STACK: Stack = { stt: 'cloud', llm: 'cloud', tts: 'cloud' };

const IN_DEVELOPMENT = process.env.NODE_ENV !== 'production';

function AppSetup() {
  useDebugMode({ enabled: IN_DEVELOPMENT });
  useAgentErrors();

  return null;
}

interface AppProps {
  appConfig: AppConfig;
}

export function App({ appConfig }: AppProps) {
  const [stack, setStack] = useState<Stack>(DEFAULT_STACK);
  const [debug, setDebug] = useState<boolean>(false);
  const [wake, setWake] = useState<WakeMode>('off');
  const [compare, setCompare] = useState<boolean>(false);

  // Keep refs to the latest values so the TokenSource callback closure always
  // reads the current dropdown values — useMemo runs once, but the callback
  // it returns can fire any time after.
  const stackRef = useRef(stack);
  stackRef.current = stack;
  const wakeRef = useRef(wake);
  wakeRef.current = wake;
  const compareRef = useRef(compare);
  compareRef.current = compare;

  const tokenSource = useMemo(() => {
    if (typeof process.env.NEXT_PUBLIC_CONN_DETAILS_ENDPOINT === 'string') {
      return getSandboxTokenSource(appConfig);
    }
    // Custom POST to /api/token with the user's stack choices in the body.
    // The server route wraps them as agent dispatch metadata so the Python
    // worker can read ctx.job.metadata on join.
    return TokenSource.custom(async () => {
      const res = await fetch('/api/token', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          stack: stackRef.current,
          wake: wakeRef.current,
          compare: compareRef.current,
        }),
      });
      if (!res.ok) throw new Error(`token fetch failed: ${res.status}`);
      return await res.json();
    });
  }, [appConfig]);

  const session = useSession(
    tokenSource,
    appConfig.agentName ? { agentName: appConfig.agentName } : undefined
  );

  return (
    <AgentSessionProvider session={session}>
      <AppSetup />
      <main className="grid h-svh grid-cols-1 place-content-center">
        <ViewController
          appConfig={appConfig}
          stack={stack}
          setStack={setStack}
          debug={debug}
          setDebug={setDebug}
          wake={wake}
          setWake={setWake}
          compare={compare}
          setCompare={setCompare}
        />
      </main>
      <StartAudioButton label="Start Audio" />
      <Toaster
        icons={{
          warning: <WarningIcon weight="bold" />,
        }}
        position="top-center"
        className="toaster group"
        style={
          {
            '--normal-bg': 'var(--popover)',
            '--normal-text': 'var(--popover-foreground)',
            '--normal-border': 'var(--border)',
          } as React.CSSProperties
        }
      />
    </AgentSessionProvider>
  );
}
