% Carregar primeiro workspace (IFOC)
load('IFOC.mat');
% Armazenar variáveis de interesse
t_IFOC = t;
Rs_vetor_IFOC = Rs_vetor;
Rr_vetor_IFOC = Rr_vetor;
ids_vetor_IFOC = ids_vetor;
Ids_ref_IFOC = Ids_ref;
Te_vetor_IFOC = Te_vetor;
Tl_IFOC = Tl;
lambda_dr_est_vetor_IFOC = lambda_dr_est_vetor;
lambda_dr_vetor_IFOC = lambda_dr_vetor;
wr_vetor_IFOC = wr_vetor;
w_ref_IFOC = w_ref;

% Limpar variáveis para evitar conflitos
clearvars -except t_IFOC Rs_vetor_IFOC Rr_vetor_IFOC ids_vetor_IFOC Ids_ref_IFOC Te_vetor_IFOC Tl_IFOC lambda_dr_est_vetor_IFOC lambda_dr_vetor_IFOC wr_vetor_IFOC w_ref_IFOC

% Carregar segundo workspace (IFOC_ANN)
load('IFOC_ANN.mat');

% Definir estilo de linha e peso
linha_estilo = {'-', '--'};
linha_estilo_2 = {'-','-','--'};
linha_peso = 2;

% Plotar e comparar ids_vetor
plotComparar2(t_IFOC, ids_vetor_IFOC, t, ids_vetor,t,Ids_ref, 'Comparação de I_{ds}', 'Tempo (s)', 'I_{ds} (A)', 'IFOC', 'IFOC - ANN','Referência', linha_estilo_2, linha_peso);
exportgraphics(gcf, 'Corrente_Ids_comp.pdf', 'ContentType', 'vector');

% % Repetir para outras variáveis
% plotComparar(t_IFOC, Rs_vetor_IFOC, t, Rs_vetor, 'Comparação de Rs_vetor', 'Tempo (s)', 'Rs_vetor', 'IFOC', 'IFOC_ANN', linha_estilo, linha_peso);
% plotComparar(t_IFOC, Rr_vetor_IFOC, t, Rr_vetor, 'Comparação de Rr_vetor', 'Tempo (s)', 'Rr_vetor', 'IFOC', 'IFOC_ANN', linha_estilo, linha_peso);

% Comparação de Te_vetor (Torque eletromagnético)
plotComparar2(t_IFOC, Te_vetor_IFOC, t, Te_vetor, t, Tl, 'Comparação de Torque Eletromagnético', 'Tempo (s)', 'Torque (N.m)', 'IFOC', 'IFOC - ANN','T_{Load}', linha_estilo_2, linha_peso);
exportgraphics(gcf, 'Torque_comp.pdf', 'ContentType', 'vector');

% Comparação de lambda_dr_est_vetor (Fluxo estimado do rotor)
plotComparar(t_IFOC, lambda_dr_est_vetor_IFOC, t_IFOC, lambda_dr_vetor_IFOC, 'Estimação do Fluxo dr - IFOC ANN', 'Tempo (s)', 'Fluxo (wb)', 'Estimado', 'Real', linha_estilo, linha_peso);
exportgraphics(gcf, 'fluxo_ifoc_comp.pdf', 'ContentType', 'vector');

% Comparação de lambda_dr_vetor (Fluxo real do rotor)
plotComparar(t, lambda_dr_est_vetor, t, lambda_dr_vetor, 'Estimação do Fluxo dr - IFOC', 'Tempo (s)', 'Fluxo (wb)', 'Estimado', 'Real', linha_estilo, linha_peso);
exportgraphics(gcf, 'fluxo_ifoc_ann_comp.pdf', 'ContentType', 'vector');

% Comparação de w_ref (Referência de velocidade)
plotComparar2(t_IFOC, wr_vetor_IFOC, t, wr_vetor,t,w_ref, 'Comparação da Velocidade Rotórica', 'Tempo (s)', 'Velocidade (rad/s)', 'IFOC', 'IFOC - ANN','Referência', linha_estilo_2, linha_peso);
exportgraphics(gcf, 'Velocidade_comp.pdf', 'ContentType', 'vector');

% Função para plotar e comparar
function plotComparar(t1, var1, t2, var2, titulo, xlabelText, ylabelText, legenda1, legenda2, estilo, peso)
    figure;
    plot(t1, var1, estilo{1}, 'LineWidth', peso, 'Color', 'k'); hold on;
    plot(t2, var2, estilo{2}, 'LineWidth', peso, 'Color', [0.5 0.5 0.5]);
    xlabel(xlabelText);
    ylabel(ylabelText);
    legend(legenda1, legenda2);
    title(titulo);
end

% Função para plotar e comparar - 2
function plotComparar2(t1, var1, t2, var2,t3,var3, titulo, xlabelText, ylabelText, legenda1, legenda2,legenda3, estilo, peso)
    figure;
    plot(t1, var1, estilo{1}, 'LineWidth', peso, 'Color', 'k'); hold on;
    plot(t2, var2, estilo{2}, 'LineWidth', peso, 'Color', [0.5 0.5 0.5]);
    plot(t3, var3, estilo{3}, 'LineWidth', peso, 'Color', [0.7 0.7 0.7]);
    xlabel(xlabelText);
    ylabel(ylabelText);
    legend(legenda1, legenda2, legenda3);
    title(titulo);
end
