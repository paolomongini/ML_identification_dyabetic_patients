> tensorboard --logdir=.\\runs\\exp_84\\paz_010 --reload_interval=5




---


optuna, toglio totalmente MSEnel calcolo

metto optuna classico

metto weight più alti

learning rate più alti

ottimizzatori diversi




---


* preprocessing dati esattamente come nella simulazione dei modelli lineari?

fare  ottimizzazioni di meal e insulina separate?

mettere la loss di non avere spike improvvisi?

(magari rispetto alla somma tutte le predizioni devono essere > 0 o di un agressive factor molto blando?)

oppure penalizzo derivate alte

magari considerare la somma dell’insulina rispetto al primo quartile

calcolare delta di glicemia rispetto G_bar o rispetto all’istante attuale?


---

```python
# train_batch_1   test_similar_to_train   insulin_impulse  meal_impulse

 #scenario	patients	delta_G_empiric	delta_G_101_linearized	delta_G_formula	delta_G_SSM  delta_t_max	FIT_101_linearized	FIT_SSM
```

8      0.1 moltiplicato

9       0.7 moltiplicato

exp 10

come 9 tranne che moltiplicato per 1 e

```python
    error > 0,
    1000 * torch.abs(error),  # errore positivo: 1000
    1.0 * torch.abs(error)   # errore negativo: pesato1
    )
```


---

11 come 4 ma loss 1000 se predizione olrìtre (come 10)


---

6**5**

config = {

\# ===== EPOCHS =====
'epochs_101': 2000,
'epochs_s1' : 2000,
'epochs_s2' : 1000,

\# ===== HYPERPARAMETERS =====
'learning_rate' : 1e-3,
'use_noise' : False,

\# ===== monotonic gain loss =====
'use_monotonic_gain_loss' : True,
'cumulative_window' : 12\\\*==4==,             # 12\\\*2.5    12\\\*4
'horizon' : 12\\\*0.5,
'type_preprocess_insulin' : '==sum==',      # 'sum'   'iob'

\# ===== monotonic gain loss =====
'use_low_pass_I' : ==True==,   # per ora implementato solo in strategy 1

}


---

75



config = {

\# ===== EPOCHS =====

'epochs_101': 2000,

'epochs_s1' : 2000,

'epochs_s2' : 1000,

\# ===== HYPERPARAMETERS =====

'learning_rate' : 1e-3,

'use_noise' : False,

\# ===== monotonic gain loss =====

'use_monotonic_gain_loss' : True,

'cumulative_window' : 12\\\*==4==,             # 12\\\*2.5    12\\\*4

'horizon' : 12\\\*0.5,

'type_preprocess_insulin' : '==iob==',      # 'sum'   'iob'

\# ===== monotonic gain loss =====

'use_low_pass_I' : ==True==,   # per ora implementato solo in strategy 1

}