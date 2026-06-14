import asyncio
import logging
import time

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    RunContext,
    WorkerOptions,
    cli,
    function_tool,
)
from livekit.plugins import cartesia, deepgram, groq, silero

logger = logging.getLogger("agent")

load_dotenv(".env.local")


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""You are AIKA, an AI Kitchen Assistant helping chefs in a restaurant.
            You respond via voice so keep responses short and clear. No formatting, no emojis.

            Your main job is managing timers. When a chef says something like "timer steak 4 minutes" or "pasta 3 minutes", use the set_timer tool.
            When they ask "how long for steak" or "check timers", use the check_timers tool.
            When they say "cancel steak timer", use the cancel_timer tool.

            Keep confirmations brief, like "Timer set. Steak, 4 minutes." or "Pasta has 2 minutes left."
            Be helpful but stay out of the way. Chefs are busy.""",
        )
        self.timers = {}  # name -> {"end_time": float, "task": asyncio.Task}

    @function_tool
    async def set_timer(self, context: RunContext, name: str, minutes: float):
        """Set a kitchen timer that will announce when done.

        Args:
            name: What the timer is for, like "steak" or "pasta"
            minutes: How many minutes to set the timer for
        """
        # Cancel existing timer with same name if any
        if name.lower() in self.timers:
            self.timers[name.lower()]["task"].cancel()

        end_time = time.time() + (minutes * 60)

        async def timer_callback():
            await asyncio.sleep(minutes * 60)
            del self.timers[name.lower()]
            session = context.session
            await session.say(f"{name} is ready!")
            logger.info(f"Timer finished: {name}")

        task = asyncio.create_task(timer_callback())
        self.timers[name.lower()] = {"end_time": end_time, "task": task}

        logger.info(f"Timer set: {name} for {minutes} minutes")
        return f"Timer set for {name}, {minutes} minutes."

    @function_tool
    async def check_timers(self, context: RunContext):
        """Check all active timers and how much time is left on each."""
        if not self.timers:
            return "No active timers."

        status = []
        now = time.time()
        for name, timer in self.timers.items():
            remaining = timer["end_time"] - now
            if remaining > 60:
                mins = int(remaining // 60)
                secs = int(remaining % 60)
                status.append(f"{name}: {mins} minutes {secs} seconds left")
            else:
                status.append(f"{name}: {int(remaining)} seconds left")

        return ". ".join(status)

    @function_tool
    async def cancel_timer(self, context: RunContext, name: str):
        """Cancel an active timer.

        Args:
            name: The name of the timer to cancel, like "steak" or "pasta"
        """
        if name.lower() in self.timers:
            self.timers[name.lower()]["task"].cancel()
            del self.timers[name.lower()]
            logger.info(f"Timer cancelled: {name}")
            return f"Timer for {name} cancelled."
        return f"No active timer for {name}."


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    await ctx.connect()

    session = AgentSession(
        stt=deepgram.STT(model="nova-3", language="multi"),
        llm=groq.LLM(model="llama-3.3-70b-versatile"),
        tts=cartesia.TTS(voice="9626c31c-bec5-4cca-baa8-f8ba9e84c8bc"),
        vad=ctx.proc.userdata["vad"],
    )

    await session.start(
        agent=Assistant(),
        room=ctx.room,
    )


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
        )
    )
