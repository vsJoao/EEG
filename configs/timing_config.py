# %% =================== Variáveis de tempo que serão utilizadas======================

# Frequencia de amostragem
sfreq = 250

# Tempo de duração total (tempo da trial)
t_trial = 7.5

# Instante inicial do intervalo de interesse em segundos
t_start = 3.5

# Instante final do intervalo de interesse em segundos
t_end = 6

# Instante iniciaç de aplicação da ICA
ica_start = 0

# Instante final de aplicação da ICA
ica_end = 7

# Tempo entre o iníncio e final das epocas
t_epoch = t_end - t_start

# Numero de samples de cada sinal
n_samples = int(t_epoch * sfreq + 1)