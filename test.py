import sklearn.neighbors as knn
import numpy as np

# ##==============================Realiza a classificação por knn============================

f = np.load('features/A01T_features.npy', allow_pickle=True).item()
features = f['lr']

for s_id, sbj_name in enumerate(f_names_test):

    # Tenta carregar as matrizes de projeção espacial desse sujeito
    try:
        W = np.load('csp/{}_Wcsp.npy'.format(f_names_train[s_id]), allow_pickle=True).item()
    except IOError:
        print('Não foram encontradas as matrizes de projeção espacial desse sujeito {}'.format(sbj_name))
        break

    # Tenta carregar o conjunto de características desse sujeito
    try:
        features = np.load('features/{}_features.npy'.format(f_names_train[s_id]), allow_pickle=True).item()
    except IOError:
        print('Não foram encontradas as matrizes de características do sujeito {}'.format(sbj_name))
        break

    total_res[sbj_name] = dict()

    for n_knn in range(20):
        total_res[sbj_name]['{}'.format(n_knn + 1)] = dict()

        for f_class in features:

            x_train = features[f_class][:, :-1]
            y_train = features[f_class][:, -1]
            KNN_model = knn.KNeighborsClassifier(n_neighbors=n_knn + 1)
            KNN_model.fit(x_train, y_train)

            f = dict()
            res = np.array([0])

            for sbj_idx in range(n_runs):

                # Carrega o arquivo raw e o conjunto de eventos referentes a ele
                raw, eve = mods.pick_file(raw_eog_fif_loc, sbj_name, sbj_idx + 1)

                for eve_idx, c_eve in enumerate(eve[:, 0] / sfreq):

                    # Salva a classe de movimento do sinal atual
                    class_mov = [i for i in e_dict if e_dict[i] == eve[eve_idx, 2]][0]

                    if class_mov == f_class[1] or class_mov == f_class[0]:

                        # Coleta uma amostra de (ica_end - ica_start) segundos para análise
                        raw_samp = raw.copy().pick('eeg').crop(tmin=c_eve + ica_start, tmax=c_eve + ica_end)

                        # Realiza a remoção de artefatos
                        raw_clean, flag = artifact_remove(raw_samp)

                        # Salva o dado em uma epoca para análise
                        curr_epoc = mods.Epochs(
                            X=raw_clean.get_data(),
                            classe=class_mov,
                            f_bank=fb_freqs,
                            e_dict=e_dict,
                            fs=sfreq
                        )

                        # Filtra o sinal nas bandas de frequencia
                        curr_epoc.filt()

                        # Calcula o vetor de caracteristicas desse sinal
                        f_temp = W[f_class].csp_feature(curr_epoc).transpose()

                        # Classifica o sinal de acordo com as caracteristicas
                        y_prediction = KNN_model.predict(f_temp)

                        # Verifica se acertou e salva
                        res = np.append(res, y_prediction == e_dict[class_mov])

            total_res[sbj_name]['{}'.format(n_knn+1)][f_class] = res.mean()
    break
