from enum import Enum


class SliceType(Enum):
    SAGITAL = "Sagital"
    CORONAL = "Coronal"
    AXIAL = "Axial"

    def __str__(self):
        return self.value
