import { Track } from 'livekit-client';
import type { AudioProcessorOptions, TrackProcessor } from 'livekit-client';

// LiveKit audio TrackProcessor that mixes a looping ambience clip INTO the
// outgoing mic track. The agent/STT receive voice + noise combined — not just
// the local speakers — so it faithfully simulates talking in a noisy kitchen.
// Runs after capture, so it bypasses the browser's NS/AEC entirely.
//
// Two gain stages:
//   send    = the SNR knob the agent hears (driven by the slider, uncapped)
//   monitor = what YOU hear locally. Tracks the slider so you get feedback that
//             it's getting louder, but capped so high send levels don't deafen.
const MONITOR_FACTOR = 0.35;
const MONITOR_CAP = 1.0;

const monitorLevel = (volume: number) => Math.min(volume * MONITOR_FACTOR, MONITOR_CAP);

export class KitchenNoiseProcessor
  implements TrackProcessor<Track.Kind.Audio, AudioProcessorOptions>
{
  name = 'kitchen-noise';
  processedTrack?: MediaStreamTrack;

  private ctx?: AudioContext;
  private voice?: MediaStreamAudioSourceNode;
  private dest?: MediaStreamAudioDestinationNode;
  private noiseNode?: MediaElementAudioSourceNode;
  private sendGain?: GainNode;
  private monitorGain?: GainNode;
  private readonly el: HTMLAudioElement;
  private volume: number;
  private readonly monitor: boolean;

  constructor(src: string, volume = 1, monitor = true) {
    this.volume = volume;
    this.monitor = monitor;
    this.el = new Audio(src);
    this.el.loop = true;
  }

  async init(opts: AudioProcessorOptions) {
    this.ctx = opts.audioContext;
    this.voice = this.ctx.createMediaStreamSource(new MediaStream([opts.track]));
    this.dest = this.ctx.createMediaStreamDestination();
    this.sendGain = this.ctx.createGain();
    this.sendGain.gain.value = this.volume;

    // voice always passes through at full level
    this.voice.connect(this.dest);

    // an element can only back one MediaElementSource for its lifetime — reuse
    if (!this.noiseNode) this.noiseNode = this.ctx.createMediaElementSource(this.el);
    this.noiseNode.connect(this.sendGain);
    this.sendGain.connect(this.dest); // -> sent to the agent at the slider level

    if (this.monitor) {
      this.monitorGain = this.ctx.createGain();
      this.monitorGain.gain.value = monitorLevel(this.volume);
      this.noiseNode.connect(this.monitorGain);
      this.monitorGain.connect(this.ctx.destination); // -> heard locally, capped
    }

    await this.el.play().catch(() => {});
    this.processedTrack = this.dest.stream.getAudioTracks()[0];
  }

  async restart(opts: AudioProcessorOptions) {
    await this.destroy();
    await this.init(opts);
  }

  async destroy() {
    this.el.pause();
    for (const n of [this.voice, this.sendGain, this.monitorGain]) {
      try {
        n?.disconnect();
      } catch {
        // already disconnected
      }
    }
    this.processedTrack = undefined;
  }

  setVolume(v: number) {
    this.volume = v;
    if (!this.ctx) return;
    const t = this.ctx.currentTime;
    this.sendGain?.gain.setTargetAtTime(v, t, 0.02); // agent hears full level
    this.monitorGain?.gain.setTargetAtTime(monitorLevel(v), t, 0.02); // you hear it ramp, capped
  }

  setClip(src: string) {
    const wasPlaying = !this.el.paused;
    this.el.src = src;
    if (wasPlaying) this.el.play().catch(() => {});
  }
}
