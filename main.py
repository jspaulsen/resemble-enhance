from pathlib import Path

import torch
import torchaudio
import typer

from resemble_enhance.enhancer.inference import denoise

app = typer.Typer()


@app.command()
def main(
    input_file: Path = typer.Argument(..., help="Path to input audio file"),
    output_file: Path = typer.Option(None, "--output", "-o", help="Path to output audio file"),
):
    """Denoise an audio file."""
    if not input_file.exists():
        typer.echo(f"Error: {input_file} does not exist", err=True)
        raise typer.Exit(1)

    if output_file is None:
        output_file = input_file.with_stem(f"{input_file.stem}_denoised")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    typer.echo(f"Using device: {device}")

    typer.echo(f"Loading {input_file}")
    dwav, sr = torchaudio.load(input_file)
    dwav = dwav.mean(dim=0)  # Convert to mono

    typer.echo("Denoising...")
    wav, new_sr = denoise(dwav, sr, device)

    typer.echo(f"Saving to {output_file}")
    torchaudio.save(output_file, wav.unsqueeze(0).cpu(), new_sr)

    typer.echo("Done!")


if __name__ == "__main__":
    app()
