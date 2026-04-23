import os
import sys
import jax
import jax.numpy as jnp
import dataclasses

from gemma_titans import Gemma3_1B_Titans, Gemma_Titans_Config

def main():
    print("Инициализация конфигурации Phase 2...")
    # Настраиваем конфиг точно так же, как в ноутбуке
    gemma_config = dataclasses.replace(
        Gemma3_1B_Titans.config,
        training_phase=2,
        titans_layer_indices=(23,), # Указываем, что 23 слой - это Titans
        titans_phase2_first_layer=23,
    )

    print("Создание модели Gemma3_1B_Titans...")
    model = Gemma3_1B_Titans(
        config=gemma_config,
        dtype=jnp.bfloat16,
        return_last_only=False,
        tokens="batch.tokens",
    )

    # Создаем dummy данные для инициализации (минимальный батч)
    key = jax.random.PRNGKey(0)
    tokens = jnp.ones((1, 10), dtype=jnp.int32)
    positions = jnp.arange(10)[None, :]
    attention_mask = jnp.ones((1, 10, 10), dtype=jnp.bool_)

    print("Вызов model.init() для генерации структуры параметров...")
    variables = model.init(
        key, 
        tokens=tokens, 
        step=0, 
        positions=positions, 
        attention_mask=attention_mask
    )
    
    params = variables.get('params', {})
    
    print("\n" + "="*40)
    print(" РЕЗУЛЬТАТЫ ПРОВЕРКИ СТРУКТУРЫ")
    print("="*40)
    
    if 'layer_23' in params:
        keys_23 = list(params['layer_23'].keys())
        print(f"Доступные компоненты внутри layer_23:\n{keys_23}\n")
        
        if 'attn' in keys_23:
            print("❌ ВНИМАНИЕ: Слой 'attn' ВСЕ ЕЩЕ ПРИСУТСТВУЕТ в layer_23!")
            print("Ключи внутри attn:", list(params['layer_23']['attn'].keys()))
            print("\nЭто значит, что исправление use_original_attn=False не сработало,")
            print("или кэши JAX (remat) продолжают использовать старую версию кода.")
        else:
            print("✅ УСПЕХ: Слой 'attn' ОТСУТСТВУЕТ в layer_23!")
            print("Модель инициализирована правильно (чистая память Titans).")
            print("Ошибка 'Source: MISSING' при загрузке чекпойнта больше не появится.")
    else:
        print("Слой layer_23 не найден в параметрах.")

if __name__ == "__main__":
    main()
