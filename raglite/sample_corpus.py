"""Built-in sample corpus: a fictional product manual for the "Nimbus 900".

Written so every golden-set question has exactly one clearly correct passage,
which makes the offline demo's answers and the eval's hit-rate verifiable by
eye without any external knowledge.
"""

from __future__ import annotations

SAMPLE_DOCS: dict[str, str] = {
    "nimbus_warranty.txt": (
        "Nimbus 900 Warranty Policy.\n"
        "The Nimbus 900 carries a standard two-year limited warranty covering "
        "manufacturing defects in materials and workmanship. The warranty period "
        "begins on the date of purchase recorded on your receipt.\n"
        "Liquid damage, dropped units, and unauthorized repairs void the warranty "
        "immediately. Extended coverage can be purchased within 30 days of the "
        "original purchase through the Nimbus support portal.\n"
        "Warranty claims require the serial number, which is printed under the "
        "battery cover and begins with the prefix NB9-."
    ),
    "nimbus_setup.txt": (
        "Nimbus 900 Quick Setup Guide.\n"
        "To power on the Nimbus 900, press and hold the side button for three "
        "seconds until the status LED glows cyan. The first boot takes about "
        "45 seconds.\n"
        "To connect Wi-Fi, open Settings, choose Network, and select your "
        "access point. The Nimbus 900 supports 2.4 GHz and 5 GHz networks but "
        "does not support 6 GHz Wi-Fi 6E.\n"
        "Factory reset: hold the side button and volume-down together for ten "
        "seconds. All user data is erased and cannot be recovered after a reset."
    ),
    "nimbus_battery.txt": (
        "Nimbus 900 Battery and Charging.\n"
        "The Nimbus 900 uses a 4,800 mAh lithium-polymer battery rated for "
        "18 hours of mixed use. A full charge from empty takes 90 minutes with "
        "the supplied 30W USB-C charger.\n"
        "For long-term battery health, keep the charge between 20 and 80 percent "
        "when possible, and avoid storing the device above 35 degrees Celsius "
        "for extended periods.\n"
        "If the battery swells, stop using the device immediately and contact "
        "support. Replacement batteries are available only through authorized "
        "service centers."
    ),
    "nimbus_troubleshooting.txt": (
        "Nimbus 900 Troubleshooting.\n"
        "If the status LED blinks red, the device has a firmware fault. Connect "
        "to power and hold the side button for 20 seconds to force a firmware "
        "recovery. Recovery takes up to five minutes and does not erase user data.\n"
        "If audio stutters over Bluetooth, move within 5 meters of the device "
        "and disable battery saver mode, which throttles the radio.\n"
        "If the touchscreen registers phantom touches, remove any third-party "
        "screen protector and clean the screen. Persistent phantom touches are "
        "a covered warranty defect."
    ),
    "nimbus_specs.txt": (
        "Nimbus 900 Technical Specifications.\n"
        "Display: 6.4-inch OLED, 120 Hz refresh rate, 2400x1080 resolution.\n"
        "Weight: 186 grams. Dimensions: 157 x 73 x 8.1 millimeters.\n"
        "Water resistance: IP68, tested to 1.5 meters of fresh water for 30 "
        "minutes. Water damage is still excluded from the warranty.\n"
        "Storage: 256 GB internal, no microSD expansion. Memory: 12 GB LPDDR5."
    ),
    "nimbus_privacy.txt": (
        "Nimbus 900 Privacy and Data Handling.\n"
        "On-device voice processing is the default: wake-word detection and "
        "transcription run locally on the Nimbus 900 and audio is never uploaded "
        "unless you explicitly start a cloud request.\n"
        "Telemetry sends anonymous crash reports every 24 hours and can be "
        "disabled in Settings, Privacy, Diagnostics.\n"
        "You can export or permanently delete all Nimbus cloud data from the "
        "privacy portal. Deletion completes within 30 days and is irreversible."
    ),
}


def seed(engine: object) -> int:
    """Ingest all sample docs into a RAGLite-like engine. Returns chunk count."""
    total = 0
    for doc_id, text in SAMPLE_DOCS.items():
        total += engine.ingest(text, doc_id=doc_id, meta={"source": "builtin-sample-corpus"})  # type: ignore[attr-defined]
    return total
