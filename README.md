# MosaicLab

MosaicLab is a local workbench for inspecting and developing a Bayer sensor raster. It keeps the original integer codes and the physical CFA phase separate from the reconstructed RGB image.

## Run locally

```sh
docker compose up --build -d
curl -fsS http://127.0.0.1:18942/api/health
```

Open [MosaicLab on localhost](http://127.0.0.1:18942). Compose binds the published port to localhost and stores the catalog and source pixels in the `solo42-mosaiclab_mosaiclab-data` named volume. Use `docker compose stop` to stop the service. Use `MOSAICLAB_PORT=18943 docker compose up -d` to publish it on a different host port.

For a direct launch with Python 3.11 or 3.12, install `requirements.txt` in a local virtual environment and call `./start.sh`. `PORT`, `MOSAICLAB_HOST`, and `MOSAICLAB_DATA_DIR` control the direct listener and catalog location.

## Start with a sample

- The **Color and neutral target** opens on first access. Switch to **Inspect** and run **Color target audit** to compare its 18 synthetic patches and download the report. The **Neutral reference** region can also set the development white balance.
- Open **Resolution field** from **Samples**. Choose the **Slanted edge** region under **Inspect** to measure the luminance MTF or the native RAW chromatic edge offset in sensor pixels. The sample is designed to show near-zero chromatic displacement.
- Open **Low-light readout** to compare two repeat captures. Choose the **Uniform readout strip** for row-periodicity and paired-noise measurements. Under **Temporal noise pair**, the whole-region σ stays visible and the **Signal step bands** table and chart split the same two captures into a shared half-stop signal ladder for all four Bayer phases. Use one of the gray bars as the region to see how σ changes close to black and at brighter levels. Pairs at or below 0.5% of the phase range or at or above 98% are excluded, and a band with too few same-site pairs records how many adjacent bins were combined to reach the threshold. A real brightness gap is never bridged; the bands stay empty and the chart breaks there.
- Open **Exposure ladder pair** to inspect clipping and headroom at each phase. Use **Compare** to inspect the overlapping registered developments.
- Select **Reference · marked sensor sites** under **Calibration references** and run **Defect mask audit**. Select a density tile or connected group in the result to set a more focused region.

All four sample workbenches and their calibration frames are generated examples. They are labeled synthetic and are intended for controlled demonstrations.

## Input and output

- **Raster inputs.** Grayscale 8-bit or 16-bit PNG, P2/P5 PGM, unsigned or nonnegative integer 2-D NPY, and binary rasters with a declared layout. A raster contains one integer code per sensor site. Binary input supports byte and 16-bit word layouts and V4L2 packed RAW10, RAW12 and RAW14. Packed row widths must be whole groups of 4, 2 and 4 sites respectively. Supply a header offset and row stride when needed. The byte-layout check reports edge codes, observed parity medians and the SHA-256 of the source bytes. Read the [Linux V4L2 packed Bayer definitions](https://docs.kernel.org/userspace-api/media/v4l/pixfmt-bayer.html).
- **Sensor metadata.** Supply the Bayer pattern at the physical origin, the sensor offset, black and white levels and an active rectangle. Black and white levels can be one value or four row-major values for the pattern. Exposure and gain are optional. Grayscale PNG image profiles and gamma tags do not change the imported integer codes.
- **Development.** Select compatible dark, flat and defect references, then use nearest-site, bilinear or edge-directed reconstruction. The workbench includes white balance, exposure, camera color matrices and tone controls. Inspect raw codes and their neighbors with a pointer, draw a region, or enter an exact raster rectangle.
- **Downloads.** Export 8- or 16-bit RGB PNG, linear 16-bit PNG, linear float NPZ, a development report, a review ZIP, or a portable raw capture. Portable raw capture ZIPs carry phase and sensor metadata with a PGM or binary pixel member. The binary writer keeps the integer codes and rejects a format whose maximum or packed group width cannot represent the requested crop or rotation. Binary rows can be padded with zeros up to the chosen alignment.
- **Workbenches.** Save an inspection or recipe, download a workbench archive, and import it into a separate workbench. Workbench archives use version 2 to retain original encoding information, and the importer accepts version 1 archives. The archive stores normalized PGM sample blocks and carries checksums, capture roles and settings.

## Limits and measurement assumptions

The application accepts at most 4,194,304 samples per frame and 8,192 sites along either side. Standalone uploads are limited to 32 MiB and workbench archives to 96 MiB. Plain-text PGM ingestion is limited to 1,048,576 samples. Bayer packing expects raster bytes with one declared layout. Camera containers, proprietary compression and live capture are outside its supported inputs.

Calibrations assume geometrically matching rasters. Pair noise assumes registered repeated scenes with compatible exposure metadata. The paired-noise signal ladder bins each pair by the mean of the two normalized codes and shares the bin edges across all four phases; it needs repeat frames whose pixels fall in the selected region, so pick one gray level or a small patch rather than the whole scene. Color audits assume accurately selected patches and a meaningful sRGB reference. The optics measurements are local estimates from a clean tilted achromatic edge. Native chromatic offsets use the mean of the two green transitions as a reference and report positive distances toward the bright side. An estimate can include effects from the optics, the sensor and the target. Read [Using Slanted Edge Analysis for Color](https://www.imaging.org/common/uploaded%20files/pdfs/Papers/1999/PICS-0-42/990.pdf).

The application deliberately caps concurrent image computations at two. Large rasters and archives take longer to process. To verify the backend source count, run `python3 -B scripts/count_backend.py --project-root . --source-root .`.
