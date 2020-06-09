import numpy as np
import tensorflow as tf

import tfrt.sources as sources

source = sources.PrecompiledSource("./data/precompiled_source_test.dat", sample_count=10)

print("loaded printout:")
for key, value in source._full_fields.items():
    print(f"{key}: {value.shape}")
    
print(f"taking some samples of 'wavelength', with sample count "
    f"{source.sample_count}."
)
for i in range(10):
    source.update()
    print(f"wavelength: {source['wavelength']}")
    
print("changing sample count...")
source.sample_count = 5    
print(f"taking some samples of 'wavelength', with sample count "
    f"{source.sample_count}."
)
for i in range(10):
    source.update()
    print(f"wavelength: {source['wavelength']}")
