# EEG-Analise
Criação de uma BCI Baseada em Imagética Motora para Iniciação Científica

Nesse projeto procura-se consctruir um modelo de Brain-Computer Interface (BCI) baseada em imagética motora, como está colocado na descrição. O banco de dados utilizado encontra-se na página da [BNCI Horizon](http://bnci-horizon-2020.eu/database/data-sets) com o título de *Four class motor imagery (001-2014)* utilizado na **BCI Competition IV**.

Os arquivos estão em formato .m (arquivo de dados do MatLab) como não encontrei nenhum módulo que conseguisse ler o arquivo .m e converter as classes do MatLab em classes lidas em python, desenvolvi o algorítmo presente no arquivo *dataset_arrangement.py* que funciona para este banco de dados em específico, salvando-os em uma pasta de forma que possam ser lidos facilmente na linguagem python, especificamente no formato *.fif* utilizado pelo módulo [MNE Tools](https://mne.tools/stable/index.html), bastante utilizado para análise de sinais bioelétricos.

(Incompleto)

## Bibliografia

ANG, Kai Keng; CHIN, Zhang Yang; ZHANG, Haihong; GUAN, Cuntai. Filter Bank Common Spatial Pattern (FBCSP) in Brain-Computer Interface. 2008 Ieee International Joint Conference On Neural Networks (ieee World Congress On Computational Intelligence), [s.l.], p.2390-2397, jun. 2008. IEEE. http://dx.doi.org/10.1109/ijcnn.2008.4634130.

FATOURECHI, Mehrdad et al. EMG and EOG artifacts in brain computer interface systems: A survey. Clinical Neurophysiology, [s.l.], v. 118, n. 3, p.480-494, mar. 2007. Elsevier BV. http://dx.doi.org/10.1016/j.clinph.2006.10.019.

GRAMFORT, Alexandre. MEG and EEG data analysis with MNE-Python. Frontiers In Neuroscience, [s.l.], v. 7, p.1-267, jul. 2013. Frontiers Media SA. http://dx.doi.org/10.3389/fnins.2013.00267.

JURCAK, Valer; TSUZUKI, Daisuke; DAN, Ippeita. 10/20, 10/10, and 10/5 systems revisited: Their validity as relative head-surface-based positioning systems. Neuroimage, [s.l.], v. 34, n. 4, p.1600-1611, fev. 2007. Elsevier BV. http://dx.doi.org/10.1016/j.neuroimage.2006.09.024.

KOLES, Z.j.. The quantitative extraction and topographic mapping of the abnormal components in the clinical EEG. Electroencephalography And Clinical Neurophysiology, [s.l.], v. 79, n. 6, p.440-447, dez. 1991. Elsevier BV. http://dx.doi.org/10.1016/0013-4694(91)90163-x.

MACHADO, Gabriel Souza. Análise do desempenho de técnicas de processamento de sinais para uma interface cérebro-computador b. 2018. 38 f. TCC (Graduação) - Curso de Engenharia de Computação, Departamento de Computação e Sistemas, Universidade Federal de Ouro Preto, João Monlevade, 2018. Disponível em: <https://www.monografias.ufop.br/handle/35400000/1214>. Acesso em: 11 fev. 2020.

NEDELCU, Elena et al. Artifact detection in EEG using machine learning. 2017 13th Ieee International Conference On Intelligent Computer Communication And Processing (iccp), Cluj-napoca, v. 13, p.77-83, set. 2017. IEEE. http://dx.doi.org/10.1109/iccp.2017.8116986. Disponível em: <https://ieeexplore.ieee.org/abstract/document/8116986>. Acesso em: 11 fev. 2020.

RAMOSER, H.; MULLER-GERKING, J.; PFURTSCHELLER, G.. Optimal spatial filtering of single trial EEG during imagined hand movement. Ieee Transactions On Rehabilitation Engineering, [s.l.], v. 8, n. 4, p.441-446, dez. 2000. Institute of Electrical and Electronics Engineers (IEEE). http://dx.doi.org/10.1109/86.895946.

SILVA, Alan Paulo Oliveira da. Uma Implementação da Análise de Componentes Independentes em Plataforma de Hardware Reconfigurável. 2010. 89 f. Dissertação (Mestrado) - Curso de Engenharia Elétrica e de Computação, Departamento de Computação e Automação, Universidade Federal do Rio Grande do Norte, Natal, 2010. Disponível em: <https://repositorio.ufrn.br/jspui/bitstream/123456789/15325/1/AlanPOS_DISSERT.pdf>. Acesso em: 11 fev. 2020.

TANGERMANN, Michael et al. Review of the BCI Competition IV. Frontiers In Neuroscience, [s.l.], v. 6, p.0-55, jul. 2012. Frontiers Media SA. http://dx.doi.org/10.3389/fnins.2012.00055. 