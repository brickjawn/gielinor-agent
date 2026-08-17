# The compute half of the agent: perception, planner, runtime.
#
# Capture and the actuator are deliberately NOT here and never will be. Both
# need host devices — a display to read and an input device to write — and the
# actuator holds the one genuinely dangerous privilege in this system. Keeping
# it outside means the process that holds the API key cannot move your mouse,
# and the process that can move your mouse holds no credential. That separation
# is the point of the split; see docs/deployment.md and docs/adr/0003.
#
# Build:  docker compose build
# Run:    docker compose up

FROM python:3.14-slim AS base

# opencv-python-headless still wants libGL and libglib at runtime even though it
# draws nothing. Installing the headless wheel without these produces an
# ImportError at `import cv2` that reads like a broken build rather than a
# missing system library.
RUN apt-get update \
    && apt-get install --no-install-recommends -y libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Non-root from here down. The agent needs no privileged operation inside the
# container, so running as root would be pure downside.
RUN useradd --create-home --uid 10001 gielinor
WORKDIR /app

# Dependencies first so a source edit doesn't invalidate the install layer.
COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN pip install --no-cache-dir .

# Assets are copied rather than mounted so the image is self-contained and
# reproducible; compose still mounts profiles read-only over the top so
# recalibration doesn't require a rebuild.
COPY assets/ ./assets/

USER gielinor

# Fails the container if perception can't construct or the profile won't load —
# both of which otherwise present as a silent no-op loop.
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "from gielinor.runtime.loop import default_perceiver; \
from gielinor.config import Settings; default_perceiver(); Settings().profile()" || exit 1

ENTRYPOINT ["python", "-m", "gielinor"]
CMD ["run"]
