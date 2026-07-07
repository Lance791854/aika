'use client';

import { useEffect, useRef, useState } from 'react';
import { LocalAudioTrack, type LocalTrackPublication, Track } from 'livekit-client';
import { useLocalParticipant } from '@livekit/components-react';
import { KitchenNoiseProcessor } from '@/lib/kitchen-noise-processor';

// Test aid: mixes a looping restaurant/kitchen ambience INTO the outgoing mic
// track, so the agent/STT actually hear voice + noise (not just your speakers).
// Volume is the SNR knob. Clips are BBC restaurant ambiences in public/.
const CLIPS = [
  { id: '1', label: 'Rest 1', src: '/restaurant-1.mp3' },
  { id: '2', label: 'Rest 2', src: '/restaurant-2.mp3' },
];

export function NoiseToggle() {
  const { localParticipant } = useLocalParticipant();
  const procRef = useRef<KitchenNoiseProcessor | null>(null);
  const ctxRef = useRef<AudioContext | null>(null);
  const [clip, setClip] = useState(CLIPS[0]);
  const [playing, setPlaying] = useState(false);
  // Gain multiplier on the noise. 1 = unity; >1 lets the ambience overpower
  // close-mic voice, which is what actually stresses a noise-robust STT.
  const [volume, setVolume] = useState(1);
  const [noMic, setNoMic] = useState(false);

  const micTrack = (): LocalAudioTrack | undefined => {
    const pub = localParticipant.getTrackPublication(Track.Source.Microphone) as
      | LocalTrackPublication
      | undefined;
    return pub?.audioTrack;
  };

  const toggle = async () => {
    const track = micTrack();
    if (!track) {
      setNoMic(true);
      return;
    }
    setNoMic(false);
    if (playing) {
      await track.stopProcessor().catch(() => {});
      procRef.current = null;
      setPlaying(false);
    } else {
      // setProcessor requires an AudioContext on the track; LiveKit doesn't
      // set one here, so provide (and reuse) our own. Click is a user gesture,
      // so creating/resuming it is allowed.
      if (!ctxRef.current) ctxRef.current = new AudioContext();
      await ctxRef.current.resume().catch(() => {});
      track.setAudioContext(ctxRef.current);
      const proc = new KitchenNoiseProcessor(clip.src, volume);
      await track.setProcessor(proc);
      procRef.current = proc;
      setPlaying(true);
    }
  };

  const onVolume = (v: number) => {
    setVolume(v);
    procRef.current?.setVolume(v);
  };

  const pickClip = (c: (typeof CLIPS)[number]) => {
    if (c.id === clip.id) return;
    setClip(c);
    procRef.current?.setClip(c.src);
  };

  // Best-effort cleanup if the panel unmounts mid-noise.
  useEffect(() => {
    return () => {
      if (procRef.current)
        micTrack()
          ?.stopProcessor()
          .catch(() => {});
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <div className="bg-background/95 fixed bottom-4 left-4 z-40 w-56 rounded-lg border p-3 shadow-md backdrop-blur">
      <div className="flex items-center justify-between">
        <h2 className="text-foreground text-xs font-semibold tracking-wider uppercase">
          Kitchen noise
        </h2>
        <button
          type="button"
          onClick={toggle}
          className={`rounded-md px-2 py-0.5 text-[10px] font-medium tracking-wider uppercase transition-colors ${
            playing ? 'bg-foreground text-background' : 'text-foreground bg-muted hover:bg-muted/70'
          }`}
        >
          {playing ? 'On' : 'Off'}
        </button>
      </div>
      {noMic ? (
        <p className="text-muted-foreground mt-2 text-[10px] leading-snug">
          Mic not live yet — unmute, then try again.
        </p>
      ) : (
        <>
          <div className="bg-muted mt-2 flex gap-1 rounded-md p-1">
            {CLIPS.map((c) => (
              <button
                key={c.id}
                type="button"
                onClick={() => pickClip(c)}
                className={`flex-1 rounded px-2 py-1 text-[10px] font-medium transition-colors ${
                  c.id === clip.id
                    ? 'bg-foreground text-background'
                    : 'text-muted-foreground hover:bg-muted/70'
                }`}
              >
                {c.label}
              </button>
            ))}
          </div>
          <div className="mt-2 flex items-center gap-2">
            <span className="text-muted-foreground text-[10px] tracking-wider uppercase">Vol</span>
            <input
              type="range"
              min={0}
              max={4}
              step={0.1}
              value={volume}
              onChange={(e) => onVolume(Number(e.target.value))}
              className="flex-1"
            />
            <span className="text-muted-foreground w-7 text-right font-mono text-[10px] tabular-nums">
              {volume.toFixed(1)}×
            </span>
          </div>
          <p className="text-muted-foreground mt-2 text-[10px] leading-snug">
            Mixed into your mic — the agent hears it too.
          </p>
        </>
      )}
    </div>
  );
}
