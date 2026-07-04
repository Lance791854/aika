import { NextResponse } from 'next/server';
import { AccessToken, type AccessTokenOptions, type VideoGrant } from 'livekit-server-sdk';
import { RoomAgentDispatch, RoomConfiguration } from '@livekit/protocol';

type StackChoice = 'cloud' | 'local' | 'gpu';
type Stack = { stt?: StackChoice; llm?: StackChoice; tts?: StackChoice };

type ConnectionDetails = {
  serverUrl: string;
  roomName: string;
  participantName: string;
  participantToken: string;
};

// NOTE: you are expected to define the following environment variables in `.env.local`:
const API_KEY = process.env.LIVEKIT_API_KEY;
const API_SECRET = process.env.LIVEKIT_API_SECRET;
const LIVEKIT_URL = process.env.LIVEKIT_URL;

// don't cache the results
export const revalidate = 0;

export async function POST(req: Request) {
  if (
    process.env.NODE_ENV !== 'development' &&
    process.env.LIVEKIT_ALLOW_INSECURE_TOKEN_API !== 'true'
  ) {
    throw new Error(
      'THIS API ROUTE IS INSECURE. DO NOT USE THIS ROUTE IN PRODUCTION WITHOUT AN AUTHENTICATION LAYER.'
    );
  }

  try {
    if (LIVEKIT_URL === undefined) {
      throw new Error('LIVEKIT_URL is not defined');
    }
    if (API_KEY === undefined) {
      throw new Error('LIVEKIT_API_KEY is not defined');
    }
    if (API_SECRET === undefined) {
      throw new Error('LIVEKIT_API_SECRET is not defined');
    }

    // Parse room config from request body.
    const body = await req.json();
    let roomConfig: RoomConfiguration;
    if (body?.room_config) {
      // Advanced clients can pass a full RoomConfiguration directly.
      roomConfig = RoomConfiguration.fromJson(body.room_config, {
        ignoreUnknownFields: true,
      });
    } else if (body?.stack || body?.wake !== undefined || body?.compare !== undefined) {
      // Frontend dropdown choices — wrap them as agent dispatch metadata so
      // they reach `ctx.job.metadata` in the Python worker. Only forward
      // valid {cloud|local|gpu} values; anything else falls back to defaults.
      const valid = (v: unknown): v is StackChoice =>
        v === 'cloud' || v === 'local' || v === 'gpu';
      const validWake = (v: unknown): v is 'off' | 'strict' =>
        v === 'off' || v === 'strict';
      const meta: Stack & { wake?: string; compare?: boolean } = {};
      if (valid(body.stack?.stt)) meta.stt = body.stack.stt;
      if (valid(body.stack?.llm)) meta.llm = body.stack.llm;
      if (valid(body.stack?.tts)) meta.tts = body.stack.tts;
      if (validWake(body.wake)) meta.wake = body.wake;
      if (typeof body.compare === 'boolean') meta.compare = body.compare;
      roomConfig = new RoomConfiguration({
        agents: [new RoomAgentDispatch({ metadata: JSON.stringify(meta) })],
      });
    } else {
      roomConfig = new RoomConfiguration();
    }

    // Generate participant token
    const participantName = 'user';
    const participantIdentity = `voice_assistant_user_${Math.floor(Math.random() * 10_000)}`;
    const roomName = `voice_assistant_room_${Math.floor(Math.random() * 10_000)}`;

    const participantToken = await createParticipantToken(
      { identity: participantIdentity, name: participantName },
      roomName,
      roomConfig
    );

    // Return connection details
    const data: ConnectionDetails = {
      serverUrl: LIVEKIT_URL,
      roomName,
      participantName,
      participantToken,
    };
    const headers = new Headers({
      'Cache-Control': 'no-store',
    });
    return NextResponse.json(data, { headers });
  } catch (error) {
    if (error instanceof Error) {
      console.error(error);
      return new NextResponse(error.message, { status: 500 });
    }
  }
}

function createParticipantToken(
  userInfo: AccessTokenOptions,
  roomName: string,
  roomConfig: RoomConfiguration | undefined
): Promise<string> {
  const at = new AccessToken(API_KEY, API_SECRET, {
    ...userInfo,
    ttl: '15m',
  });
  const grant: VideoGrant = {
    room: roomName,
    roomJoin: true,
    canPublish: true,
    canPublishData: true,
    canSubscribe: true,
  };
  at.addGrant(grant);

  if (roomConfig) {
    at.roomConfig = roomConfig;
  }

  return at.toJwt();
}
