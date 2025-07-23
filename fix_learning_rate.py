# Verificar y arreglar el learning rate bug
import re

# Leer trainer.py actual
with open('poker_bot/core/trainer.py', 'r') as f:
    content = f.read()

print("🔍 Buscando aplicación de learning rate en _cfr_step_pure...")

# Buscar la función _cfr_step_pure
cfr_step_start = content.find('def _cfr_step_pure(')
if cfr_step_start == -1:
    print("❌ No se encontró _cfr_step_pure")
    exit()

# Encontrar el final de la función (próxima función o final del archivo)
next_def = content.find('\ndef ', cfr_step_start + 1)
if next_def == -1:
    cfr_step_content = content[cfr_step_start:]
else:
    cfr_step_content = content[cfr_step_start:next_def]

print(f"📝 Contenido de _cfr_step_pure (últimas 20 líneas):")
lines = cfr_step_content.split('\n')
for i, line in enumerate(lines[-20:]):
    print(f"  {len(lines)-20+i:2}: {line}")

# Buscar menciones de learning_rate
learning_rate_lines = []
for i, line in enumerate(lines):
    if 'learning_rate' in line.lower():
        learning_rate_lines.append((i, line.strip()))

print(f"\n🎯 Líneas con 'learning_rate':")
for line_num, line in learning_rate_lines:
    print(f"  {line_num}: {line}")

# Buscar la línea específica donde se aplica
regret_update_lines = []
for i, line in enumerate(lines):
    if 'regret_updates' in line and '*' in line:
        regret_update_lines.append((i, line.strip()))

print(f"\n🔧 Líneas donde se multiplican regret_updates:")
for line_num, line in regret_update_lines:
    print(f"  {line_num}: {line}")

if not regret_update_lines:
    print(f"\n❌ BUG CONFIRMADO: No se está aplicando learning_rate a regret_updates")
    print(f"   Necesitamos agregar: regret_updates = regret_updates * learning_rate")
else:
    print(f"\n✅ Learning rate se aplica en las líneas mostradas arriba")

# Verificar el valor del learning rate en config
print(f"\n📊 Verificando learning_rate en config...")
with open('config/training_config.yaml', 'r') as f:
    config_content = f.read()

import re
lr_match = re.search(r'learning_rate:\s*([0-9.]+)', config_content)
if lr_match:
    lr_value = float(lr_match.group(1))
    print(f"  Config learning_rate: {lr_value}")
    if lr_value == 0.02:
        print(f"  ✅ Learning rate correcto en config")
    else:
        print(f"  ⚠️ Learning rate inesperado: {lr_value}")
else:
    print(f"  ❌ No se encontró learning_rate en config")