"""Condizioni visive di una ripresa: solo dati, nessuna dipendenza da MuJoCo.

Servono alla domain randomization del visore. Chi le campiona (il modulo di
addestramento) non conosce MuJoCo; chi le applica (`Simulator`) non sa da quale
distribuzione arrivano. Il default di ogni campo e' "nessun cambiamento", cosi'
le scene che non le usano restano identiche a prima.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math


GROUND_KINDS = ("checker", "flat", "noise")


@dataclass(frozen=True)
class CameraPerturbation:
    """Spostamento del rig stereo rispetto alla posa nominale della scena.

    Il rig resta rettificato: le due camere si muovono insieme e la baseline
    non cambia. `fovy_deg=None` lascia il campo visivo nominale.
    """

    position_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    target_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    roll_deg: float = 0.0
    fovy_deg: float | None = None

    def __post_init__(self):
        for name in ("position_offset", "target_offset"):
            value = getattr(self, name)
            if len(value) != 3 or not all(math.isfinite(v) for v in value):
                raise ValueError(f"{name} deve essere una terna finita")
        if not math.isfinite(self.roll_deg):
            raise ValueError("roll_deg non valido")
        if self.fovy_deg is not None and not 1.0 <= self.fovy_deg <= 170.0:
            raise ValueError("fovy_deg deve essere in [1, 170]")


@dataclass(frozen=True)
class LightingConditions:
    """Luce principale direzionale e headlight della camera.

    `direction` e' la direzione di propagazione (dalla luce verso la scena);
    `ambient` e' la luce ambiente globale, portata dall'headlight.
    """

    direction: tuple[float, float, float]
    diffuse: float
    ambient: float
    headlight: float
    cast_shadow: bool = True

    def __post_init__(self):
        if len(self.direction) != 3 or not all(math.isfinite(v) for v in self.direction):
            raise ValueError("direction deve essere una terna finita")
        if math.hypot(*self.direction) <= 0:
            raise ValueError("direction non puo' essere nulla")
        for name in ("diffuse", "ambient", "headlight"):
            value = getattr(self, name)
            if not 0.0 <= value <= 2.0:
                raise ValueError(f"{name} deve essere in [0, 2]")


@dataclass(frozen=True)
class GroundAppearance:
    """Aspetto del terreno.

    - `checker`: la texture originale a scacchi;
    - `flat`: tinta unita di grigio `base_gray`;
    - `noise`: macchie casuali attorno a `base_gray`, ampiezza `contrast`,
      grana `grain_px` pixel di texture, generate da `seed`.

    `reflectance=None` lascia la riflessione nominale del pavimento (0.2): un
    terreno reale come sabbia o terra non riflette gli oggetti, quindi la
    randomizzazione la puo' abbassare.
    """

    kind: str = "checker"
    base_gray: float = 0.5
    contrast: float = 0.0
    grain_px: int = 4
    seed: int = 0
    reflectance: float | None = None

    def __post_init__(self):
        if self.kind not in GROUND_KINDS:
            raise ValueError(f"kind deve essere uno di {GROUND_KINDS}")
        if not 0.0 <= self.base_gray <= 1.0 or not 0.0 <= self.contrast <= 1.0:
            raise ValueError("base_gray e contrast devono essere in [0, 1]")
        if self.grain_px < 1 or self.seed < 0:
            raise ValueError("grain_px deve essere positivo e seed non negativo")
        if self.reflectance is not None and not 0.0 <= self.reflectance <= 1.0:
            raise ValueError("reflectance deve essere in [0, 1]")


@dataclass(frozen=True)
class BackgroundAppearance:
    """Sfondo oltre il terreno (lo skybox MuJoCo), in scala di grigio.

    Con la camera inclinata il bordo del terreno e il cielo entrano
    nell'inquadratura: se restano sempre lo stesso gradiente blu, il detector
    puo' usarli come riferimento fisso che nel mondo reale non esiste.
    """

    base_gray: float = 0.3
    contrast: float = 0.0
    seed: int = 0

    def __post_init__(self):
        if not 0.0 <= self.base_gray <= 1.0 or not 0.0 <= self.contrast <= 1.0:
            raise ValueError("base_gray e contrast devono essere in [0, 1]")
        if self.seed < 0:
            raise ValueError("seed non puo' essere negativo")


@dataclass(frozen=True)
class VisualConditions:
    """Tutto cio' che cambia l'immagine senza cambiare la fisica.

    `object_rgb` assegna un colore a singole istanze; quelle non elencate
    tengono il colore del dataset. Un campo `None` lascia il valore nominale.
    """

    camera: CameraPerturbation | None = None
    lighting: LightingConditions | None = None
    ground: GroundAppearance | None = None
    background: BackgroundAppearance | None = None
    object_rgb: dict[str, tuple[float, float, float]] = field(default_factory=dict)

    def __post_init__(self):
        for instance_id, rgb in self.object_rgb.items():
            if len(rgb) != 3 or not all(0.0 <= value <= 1.0 for value in rgb):
                raise ValueError(f"Colore non valido per {instance_id}: {rgb}")
