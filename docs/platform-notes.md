# Platform notes: Wayland and macOS

Both platforms are first-class; neither is the primary. Everything here sits
behind the `CaptureBackend` and `Actuator` protocols in
[`contracts.py`](../src/gielinor/contracts.py), and the pure-software
`ReplayCapture` / `DryRunActuator` pair is what CI and every test actually run.
Nothing in this document should ever be required to run the test suite.

Details below reflect the state of these APIs as of August 2026. Portal
backends and TCC behaviour both move; verify against your own system rather
than trusting a document.

---

## Shared preconditions

Two requirements apply on both platforms, and both come from
[ADR 0002](adr/0002-glyph-ocr-over-general-ocr.md).

**Captured pixels must be rendered pixels, 1:1.** Exact glyph matching is
correct because the renderer does no anti-aliasing; any resampling anywhere in
the capture path destroys that property and turns a deterministic decode into a
pile of near-misses. The realistic culprits are fractional output scaling on
Wayland and point-versus-pixel confusion on Retina displays. Both are covered
below.

**The client runs at a fixed 765x503 viewport with any client-side smoothing or
HD mode disabled.** All coordinates and colours live in
`assets/profiles/*.toml`, so retargeting is a data change — but the profile is
only valid for the resolution it was calibrated at.

The startup self-test that enforces these matches a known UI sprite byte-for-byte
and refuses to run on mismatch. When it fails, the cause is almost always one of
the two items above.

---

## Wayland (Linux)

### Capture: why the obvious approach does not work

Under X11 any client could call `XGetImage` on the root window and read the
whole screen, including other applications' windows. That was a security hole,
and Wayland closed it deliberately: **a Wayland client cannot read another
client's surface.** `mss`, `pyautogui`, and `PIL.ImageGrab` therefore do not
work as they do on X11.

The confusing part is that they do not fail cleanly. 2009scape is a Java client,
and OpenJDK's native Wayland support (Project Wakefield) is still in progress,
so in practice the client runs as an **XWayland** surface. That means X11
tooling can appear to partially work — returning black frames, stale frames, or
only XWayland content depending on the compositor. A partial success is worse
than a clean failure, because it survives a smoke test and breaks later. Do not
build on it, and expect it to break entirely the day OpenJDK ships a native
Wayland backend.

### Capture: the sanctioned path

`xdg-desktop-portal`'s ScreenCast interface, with frames delivered over
PipeWire. The D-Bus sequence on `org.freedesktop.portal.ScreenCast`:

1. `CreateSession`
2. `SelectSources` — set `types` (1 = monitor, 2 = window, 4 = virtual),
   `cursor_mode` (1 = hidden, 2 = embedded, 4 = metadata), and `persist_mode`
   (2 = persistent).
3. `Start` — the compositor shows the user a picker dialog. Returns a list of
   streams as `(node_id, properties)`, and a `restore_token` if persistence was
   requested.
4. `OpenPipeWireRemote` — returns a file descriptor.
5. Connect to PipeWire with that fd, connect to the returned node id, negotiate
   a format, and start receiving buffers.

Steps 1-4 cost a few hundred milliseconds plus however long the user takes in
the dialog. They happen once at startup and are not part of the per-frame budget
in [latency-budget.md](latency-budget.md).

**Use `restore_token` and `persist_mode=2`** (ScreenCast portal version 4 and
later). Without it the user gets a picker dialog on every launch, which makes
the agent unusable in practice. Store the token in the config directory; the
portal may return a *new* token on each `Start`, so always write back whatever
you were handed rather than assuming the old one stays valid.

**Ask for shared-memory buffers, not DMA-BUF.** If the stream negotiates
DMA-BUF, reading it from CPU means mapping GPU memory, and uncached reads over
PCIe cost 5-15 ms per frame — several times the acquisition budget. Treat a
DMA-BUF negotiation as a configuration problem and log it loudly rather than
silently eating the cost.

### Compositor differences that actually matter

| Compositor | Portal backend | Monitor capture | Window capture | RemoteDesktop (input) |
| --- | --- | --- | --- | --- |
| Sway / wlroots | `xdg-desktop-portal-wlr` | yes | no | no |
| GNOME / Mutter | `xdg-desktop-portal-gnome` | yes | yes | yes |
| KDE / KWin | `xdg-desktop-portal-kde` | yes | yes | yes |

The wlroots backend is screencast-only and captures whole outputs, not
individual windows. Since we have to support it, **capture the monitor on every
compositor and locate the client window inside the frame**, rather than taking
a per-window capture where it is available. That is uniform, and it suits us
anyway: we need to find the viewport by anchor sprite regardless, and an
output-level capture is the easiest place to guarantee the 1:1 pixel
precondition.

**Fractional scaling breaks glyph matching.** At output scale 1.25 or 1.5, the
client is rendered at a non-integer scale and resampled somewhere in the
pipeline. Run the game on an output set to integer scale 1. If the desktop needs
fractional scaling elsewhere, give the client its own output or its own
workspace on an unscaled one.

**`XDG_CURRENT_DESKTOP` must be correct**, or the portal picks the wrong backend
and `Start` fails in ways that do not name the real cause. On Sway this is the
single most common setup failure.

### Wayland setup steps

```bash
# 1. Install. Package names vary by distro; these are the Arch names.
sudo pacman -S pipewire wireplumber xdg-desktop-portal
sudo pacman -S xdg-desktop-portal-wlr    # Sway/wlroots
# or  xdg-desktop-portal-gnome           # GNOME
# or  xdg-desktop-portal-kde             # KDE

# 2. Make sure the portal sees the session. On Sway, add to your config:
#      exec dbus-update-activation-environment --systemd \
#           WAYLAND_DISPLAY XDG_CURRENT_DESKTOP
#    Skipping this is the most common cause of a portal that "does nothing".

# 3. Confirm the right backend is registered.
ls /usr/share/xdg-desktop-portal/portals/
busctl --user introspect org.freedesktop.portal.Desktop \
       /org/freedesktop/portal/desktop | grep -i screencast

# 4. Confirm PipeWire is up and can see a screencast node once a session starts.
systemctl --user status pipewire wireplumber
pw-cli info all | head -40

# 5. Load uinput at boot and grant it to a dedicated group (not your login shell).
echo uinput | sudo tee /etc/modules-load.d/uinput.conf
sudo groupadd -f gielinor-input
printf '%s\n' 'KERNEL=="uinput", SUBSYSTEM=="misc", GROUP="gielinor-input", MODE="0660", OPTIONS+="static_node=uinput"' \
  | sudo tee /etc/udev/rules.d/60-gielinor-uinput.rules
sudo udevadm control --reload-rules && sudo udevadm trigger
sudo modprobe uinput
ls -l /dev/uinput        # expect crw-rw---- root gielinor-input

# 6. Give the actuator — and only the actuator — that group.
#    Adding your user to it grants every process you run, which is exactly
#    what ADR 0003 argues against. Prefer a systemd user unit:
#      [Service]
#      SupplementaryGroups=gielinor-input
#    A separate uid would make this a real privilege boundary; v1 does not.

# 7. Set the game's output to integer scale and start the client at 765x503.
```

### Input: `/dev/uinput`

`uinput` creates a virtual input device at the **evdev** layer, below the
compositor. libinput enumerates it like any physical keyboard or mouse, so it
works identically on every compositor. That portability is why it is the default
choice despite the setup cost.

Practical details that are easy to get wrong:

- **Register an absolute pointing device**, not a relative one: `UI_SET_EVBIT`
  with `EV_ABS`, `UI_SET_ABSBIT` for `ABS_X`/`ABS_Y`, and `absinfo` maxima set
  to the output resolution. Relative (`REL_X`/`REL_Y`) motion has to accumulate
  deltas and fight pointer acceleration, which is a genuine source of misclicks.
  The cost is that an absolute device maps onto the whole desktop layout, so
  multi-monitor setups need that mapping accounted for; a single output is
  simplest.
- **Sleep 100-200 ms after `UI_DEV_CREATE`.** udev and libinput need time to
  enumerate the new device. Events sent before enumeration completes are
  silently dropped, which presents as "the first click never works".
- **Emit `SYN_REPORT` after every batch.** Without it, nothing is delivered.
- **Destroy the device on exit** (`UI_DEV_DESTROY`). Orphaned virtual devices
  accumulate across crashes and confuse later debugging.

### Input: alternatives, and when to prefer them

`org.freedesktop.portal.RemoteDesktop` is arguably the *correct* Wayland answer:
`NotifyPointerMotionAbsolute`, `NotifyPointerButton`, and `NotifyKeyboardKeycode`,
with no root, no udev rule, and no kernel module. It can be linked to the same
ScreenCast session so pointer coordinates are expressed in stream space, which
removes a whole class of coordinate-mapping bugs.

It is not the default here for one reason: `xdg-desktop-portal-wlr` does not
implement it, so Sway and other wlroots compositors have no backend for it. The
sensible arrangement is uinput as the portable path with RemoteDesktop preferred
where available.

`ydotool` is a daemon wrapper around uinput and adds a process without removing
a permission. `wtype` needs `zwp_virtual_keyboard_v1`, which GNOME does not
implement. Neither is a better default.

### CachyOS and other Arch derivatives

This is the current target machine, so the specifics are worth writing down.

**Install PyGObject from `pacman`, not `pip`.** The pip wheel builds against
system GLib/GObject headers and breaks in confusing ways on a rolling release
whose libraries have moved since the wheel was built. The distro package is
already matched to the system:

```bash
sudo pacman -S python-gobject gst-python gst-plugin-pipewire \
               gstreamer gst-plugins-base
# KDE Plasma is CachyOS's default; swap the backend if you run something else.
sudo pacman -S xdg-desktop-portal xdg-desktop-portal-kde
```

Then create the project venv with `--system-site-packages` so it can see
`gi`, or the portal path will import-fail inside the venv while working fine
outside it — a genuinely annoying twenty minutes.

**Arch ships a newer Python than this project's floor.** `pyproject.toml`
declares `>=3.11` and CI tests 3.11 and 3.13, so a current Arch Python is
covered. If a dependency has no wheel for the very newest interpreter yet,
`pacman -S python311` and point the venv at it rather than fighting the build.

**The Java client will almost certainly run under XWayland, and that is the
thing to check first.** OpenJDK's native Wayland support is still experimental
(Project Wakefield), so an AWT/Swing client like 2009scape's gets an XWayland
surface. Three consequences, in descending order of how much time they can cost:

1. **XWayland scaling is a separate setting from Wayland output scaling, and it
   is the most likely way exact glyph matching silently dies.** A fractionally
   scaled XWayland surface is rendered at one size and resampled to another; the
   drop shadow softens, the glyph edges pick up intermediate values, and OCR
   accuracy collapses to roughly Tesseract's. Confirm the client is rendering
   1:1 before building a font atlas from it — ADR 0002 treats this as a startup
   precondition rather than an assumption for exactly this reason.
2. Because it *is* an X11 client, X11 tooling can see it. That makes
   `xwininfo`/`xdotool` useful for confirming the true client-area geometry
   during Phase 1 calibration, even though the agent itself captures through the
   portal.
3. `uinput` is unaffected. It injects at the kernel level, so events reach the
   compositor and get routed to the focused surface normally — XWayland or not.
   This is the main reason to prefer uinput over the RemoteDesktop portal here.

**CachyOS's scheduler tuning is a mild plus.** The BORE/sched-ext work reduces
latency jitter under load, which is helpful for a loop with a 50 ms frame
budget. Do not count on it to fix a real budget overrun — the fix for that is in
the code — but it means less noise in the timing numbers.

---

## macOS

### Capture: ScreenCaptureKit

ScreenCaptureKit (macOS 12.3+) is the supported API. `CGWindowListCreateImage`
was deprecated in macOS 14 and had been gated behind Screen Recording since
10.15; `SCScreenshotManager` (14.0+) is for one-shot stills, not a 20 Hz stream.

`pyobjc-framework-ScreenCaptureKit` exposes the framework to Python, alongside
`pyobjc-framework-Quartz` and `pyobjc-framework-CoreMedia` for the pixel-buffer
and timestamp handling. Per `SPEC.md` §6, each of these needs a one-line
justification in the module docstring of whatever imports it.

The shape:

1. `SCShareableContent` to enumerate displays and windows.
2. An `SCContentFilter` — a display filter is the right choice, for the same
   reason as on Wayland: capture the display and locate the client inside it.
3. An `SCStreamConfiguration`: `pixelFormat = kCVPixelFormatType_32BGRA`,
   `minimumFrameInterval = CMTimeMake(1, 20)` to cap at 20 Hz, a `queueDepth` of
   5-8, `showsCursor` off, audio off.
4. An `SCStream` with an output handler implementing
   `stream:didOutputSampleBuffer:ofType:`.

### Four gotchas, all of which produce a plausible-looking wrong image

**Idle frames.** ScreenCaptureKit only delivers a buffer when content changes.
Buffers carry an `SCStreamFrameInfoStatus` attachment of `.complete`, `.idle`,
`.blank`, `.suspended`, or `.started`, and an `.idle` buffer may have no usable
surface at all. A static screen therefore produces a stream of `.idle` frames. A
grab loop must treat `.idle` as *reuse the last frame*, not as *no frame* and
certainly not as *black frame*. This is the usual cause of "my capture is
black".

**Row padding.** `CVPixelBufferGetBytesPerRow` is generally **not**
`width * 4` — IOSurfaces are row-aligned, often to 64 bytes. Building a numpy
array as `frombuffer(...).reshape(h, w, 4)` produces a progressively sheared
image. Reshape to `(h, bytes_per_row // 4, 4)` and slice `[:, :w, :]`.

**Points versus pixels.** `SCDisplay.width`/`.height` are in points. Setting the
stream configuration from them yields a half-resolution capture on a 2x Retina
display, and the resampling that follows breaks the exact-glyph precondition
from [ADR 0002](adr/0002-glyph-ocr-over-general-ocr.md). Multiply by the backing
scale factor — `NSScreen.backingScaleFactor`, or the ratio of
`CGDisplayModeGetPixelWidth` to `CGDisplayModeGetWidth`.

**Buffer lifetime.** Lock with
`CVPixelBufferLockBaseAddress(buf, kCVPixelBufferLock_ReadOnly)`, copy the bytes
out, then unlock. The underlying IOSurface is recycled; a numpy array that
merely wraps the pointer will show a different frame moments later.

For frame timestamps, `CMSampleBufferGetPresentationTimeStamp` is on the host
time base (`CMClockGetHostTimeClock`), which is `mach_absolute_time` — the same
base as `time.monotonic()` on macOS. That makes `Frame.captured_at` mean what
its docstring claims rather than "when Python noticed". See
[latency-budget.md](latency-budget.md).

### Permissions: two separate TCC grants

They are distinct, they are requested at different moments, and having one does
not imply the other.

| Permission | TCC service | Needed for | Settings pane |
| --- | --- | --- | --- |
| Screen Recording | `kTCCServiceScreenCapture` | ScreenCaptureKit | Privacy & Security > Screen & System Audio Recording |
| Accessibility | `kTCCServiceAccessibility` | `CGEventPost` reaching other apps | Privacy & Security > Accessibility |

**Input Monitoring** (`kTCCServiceListenEvent`) is for *listening* to events via
an event tap. We only post events, so we do not need it. Noted here so it is not
granted out of habit — see [ADR 0003](adr/0003-untrusted-planner.md) on not
over-granting.

Check Accessibility programmatically with `AXIsProcessTrustedWithOptions` and
fail at startup with a clear message rather than discovering it when the first
click silently does nothing.

### The responsible-process trap

TCC attributes a grant to the **responsible process**, not to the script. Run a
Python script from Terminal and macOS prompts for *Terminal*, and granting it
grants every program anyone ever runs from a terminal on that machine. That is
precisely the over-granting ADR 0003 objects to, and it is the default outcome
if you do nothing.

The narrower arrangement is to bundle the actuator as its own signed `.app` so it
carries its own TCC identity, and grant only that.

Two related failure modes to expect:

- **Changing the binary invalidates the grant.** Rebuilding, re-signing, or
  upgrading Python through Homebrew changes the code signature. The Settings
  entry remains visible but stops working — a stale grant that looks correct.
  Remove and re-add it, or `tccutil reset ScreenCapture <bundle-id>`.
- **Interpreters are not interchangeable.** A python.org framework build, a
  Homebrew python, and a pyenv build are three different identities. Switching
  between them silently loses permissions, and the symptom is a black capture or
  a click that does nothing, not an error.

### Input: CGEvent

`CGEventPost(kCGHIDEventTap, event)` injects at the HID level, which is what
behaves like a real mouse as far as the game is concerned. `CGEventPostToPid`
targets a single process and is more contained but less faithful.

- Move the cursor with a `CGEventCreateMouseEvent` move event rather than
  `CGWarpMouseCursorPosition` alone — a warp repositions the cursor without
  generating an event, and applications that track motion will not see it.
- If you do warp, call `CGEventSourceSetLocalEventsSuppressionInterval(source, 0)`.
  The default suppresses local events for about a quarter second after a warp,
  which reads as unresponsiveness.
- **Secure Input** (`EnableSecureEventInput`) blocks synthetic keyboard events
  entirely while a password field is focused, including at a `sudo` prompt in
  Terminal. This is the OS partially mitigating the threat in ADR 0003 for us.
  It is a useful backstop and not a control we own, so it does not change the
  argument for keeping the verb set closed.

### macOS setup steps

```bash
# 1. macOS 12.3 or later; 14+ preferred.
sw_vers

# 2. Install the framework bindings.
pip install pyobjc-framework-ScreenCaptureKit pyobjc-framework-Quartz \
            pyobjc-framework-CoreMedia

# 3. Trigger the Screen Recording prompt by attempting one capture, then
#    approve it in System Settings > Privacy & Security >
#    Screen & System Audio Recording. Restart the process afterwards; the
#    grant is not picked up mid-run.

# 4. Grant Accessibility in System Settings > Privacy & Security >
#    Accessibility. Add the actuator bundle, not your terminal, if you have
#    built one.

# 5. Verify both, and fail loudly if either is missing.
python -c "from ApplicationServices import AXIsProcessTrusted; print(AXIsProcessTrusted())"

# 6. If a grant appears present but does not work, it is stale. Reset it:
tccutil reset ScreenCapture   <bundle-id-or-blank-for-all>
tccutil reset Accessibility   <bundle-id-or-blank-for-all>

# 7. Run the client at 765x503 on a display whose scale factor you have
#    accounted for in the stream configuration.
```

---

## Quick platform comparison

| Concern | Wayland | macOS |
| --- | --- | --- |
| Capture API | xdg-desktop-portal ScreenCast + PipeWire | ScreenCaptureKit |
| Python access | D-Bus client + PipeWire (GStreamer `pipewiresrc` is the pragmatic route) | `pyobjc-framework-ScreenCaptureKit` |
| Per-launch user prompt | Yes, unless `restore_token` is used | Once, then persistent per identity |
| Frames when nothing changes | Delivered | Delivered as `.idle`, must reuse last |
| Pixel-fidelity risk | Fractional output scaling | Retina points versus pixels |
| Input mechanism | `/dev/uinput` (portable) or RemoteDesktop portal (GNOME/KDE) | `CGEventPost` |
| Input permission | udev rule on `/dev/uinput` | Accessibility TCC grant |
| Permission granularity | Per group, per process if run under a systemd unit | Per code-signing identity |
| Most common failure | `XDG_CURRENT_DESKTOP` unset, wrong portal backend | Stale TCC grant after an interpreter change |
