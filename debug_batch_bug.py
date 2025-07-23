# Debug específico del batch accumulation bug
import re

print("🔍 DEBUGGING BATCH ACCUMULATION BUG:")
print("=" * 50)

# Leer el código de _cfr_step_pure
with open('poker_bot/core/trainer.py', 'r') as f:
    content = f.read()

# Encontrar la función _cfr_step_pure
cfr_start = content.find('def _cfr_step_pure(')
cfr_end = content.find('\ndef ', cfr_start + 1)
if cfr_end == -1:
    cfr_content = content[cfr_start:]
else:
    cfr_content = content[cfr_start:cfr_end]

lines = cfr_content.split('\n')

print("🔍 Buscando líneas críticas de batch processing...")

# Buscar líneas con batch_regret_updates
batch_lines = []
sum_lines = []
mean_lines = []

for i, line in enumerate(lines):
    if 'batch_regret_updates' in line:
        batch_lines.append((i, line.strip()))
    if 'jnp.sum(' in line and 'regret' in line:
        sum_lines.append((i, line.strip()))
    if 'jnp.mean(' in line and 'regret' in line:
        mean_lines.append((i, line.strip()))

print(f"\n📊 Líneas con 'batch_regret_updates':")
for line_num, line in batch_lines:
    print(f"  {line_num:3}: {line}")

print(f"\n🧮 Líneas con 'jnp.sum' + regret:")
for line_num, line in sum_lines:
    print(f"  {line_num:3}: {line}")

print(f"\n📈 Líneas con 'jnp.mean' + regret:")
for line_num, line in mean_lines:
    print(f"  {line_num:3}: {line}")

# Buscar específicamente la línea problemática
problem_line = None
for i, line in enumerate(lines):
    if 'regret_updates = jnp.sum(batch_regret_updates' in line:
        problem_line = (i, line.strip())
        break

if problem_line:
    line_num, line_content = problem_line
    print(f"\n🚨 LÍNEA PROBLEMÁTICA ENCONTRADA:")
    print(f"  {line_num:3}: {line_content}")
    print(f"\n🔧 FIX SUGERIDO:")
    fixed_line = line_content.replace('jnp.sum(', 'jnp.mean(')
    print(f"  {line_num:3}: {fixed_line}")
    print(f"\n💡 Explicación:")
    print(f"  - jnp.sum() acumula regrets de TODOS los juegos del batch")
    print(f"  - jnp.mean() promedia regrets (comportamiento CFR correcto)")
    print(f"  - Con batch_size=128, sum() es 128x más grande que mean()")
else:
    print(f"\n❓ No se encontró la línea específica, buscando patrones alternativos...")
    
    # Buscar cualquier suma de regrets
    for i, line in enumerate(lines):
        if 'jnp.sum(' in line and ('regret' in line or 'update' in line):
            print(f"  {i:3}: {line.strip()}")

# Análisis matemático del bug
print(f"\n🧮 Análisis matemático:")
print(f"  Batch size: 128")
print(f"  Ratio observado: 2.635")
print(f"  Ratio/batch_size: {2.635/128:.4f}")
print(f"  √(Ratio): {2.635**0.5:.3f}")

print(f"\n💭 Hipótesis:")
print(f"  - Si el info set aparece en promedio 2-3 juegos por batch")
print(f"  - Y se usa jnp.sum() en lugar de jnp.mean()")
print(f"  - Entonces factor = 2.6x explicaría el bug")