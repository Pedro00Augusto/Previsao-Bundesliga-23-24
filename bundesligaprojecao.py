# -*- coding: utf-8 -*-
#Comando Console Spyder python -m spyder_kernels.console
import pandas as pd
import requests
from scipy.stats import poisson

dados = requests.get('https://pt.wikipedia.org/wiki/Bundesliga_de_2023–24')
dados_tabelas = pd.read_html(dados.text)

#Tabela de Classificação Atual
classificacao = dados_tabelas[8]
#Tabela de Jogos 
jogos = dados_tabelas[11]    

nomeTimes = list(jogos['Casa \ Fora'])
siglaTimes = list(jogos.columns)
siglaTimes.pop(0)

dicioTimes = dict(zip(siglaTimes, nomeTimes))

jogos_ajustada = jogos.set_index('Casa \ Fora')
jogos_ajustada = jogos_ajustada.unstack()
jogos_ajustada = jogos_ajustada.reset_index()
jogos_ajustada = jogos_ajustada.rename(columns={"level_0": "Fora","Casa \ Fora": "Casa", 0: "Resultado"})

def trocarSigla(linha):
    sigla = linha["Fora"]
    nome = dicioTimes[sigla]
    return nome

jogos_ajustada["Fora"] = jogos_ajustada.apply(trocarSigla, axis = 1)
jogos_ajustada = jogos_ajustada[jogos_ajustada["Fora"]!=jogos_ajustada["Casa"]]
jogos_ajustada["Resultado"] = jogos_ajustada["Resultado"].fillna("A Jogar")

jogos_feitos = jogos_ajustada[jogos_ajustada["Resultado"].str.contains("–")]
jogos_faltantes = jogos_ajustada[~jogos_ajustada["Resultado"].str.contains("–")]

  
jogos_feitos[["gols_casa", "gols_fora"]] = jogos_feitos["Resultado"].str.split("–", expand=True)
jogos_feitos = jogos_feitos.drop(columns=["Resultado"])
jogos_feitos["gols_casa"] = jogos_feitos["gols_casa"].astype(int) 
jogos_feitos["gols_fora"] = jogos_feitos["gols_fora"].astype(int)

media_gfc = jogos_feitos.groupby("Casa").mean(numeric_only=True)
media_gfc = media_gfc.rename(columns={"gols_casa": "gols_feitos_casa", "gols_fora": "gols_sofridos_casa"})

media_gff = jogos_feitos.groupby("Fora").mean(numeric_only=True)
media_gff = media_gff.rename(columns={"gols_casa": "gols_sofridos_fora", "gols_fora": "gols_feitos_fora"})

estatisticas = media_gfc.merge(media_gff, left_index = True, right_index = True) 
estatisticas = estatisticas.reset_index().rename(columns={"Casa": "Time"})

tabela_erro = jogos_feitos

def criar_tabela_erro(linha):
    timeHome = linha["Casa"]
    timeAway = linha["Fora"]
    golsHome = linha["gols_casa"]
    golsAway = linha['gols_fora']

    gfc = estatisticas.loc[estatisticas["Time"]==timeHome, "gols_feitos_casa"].iloc[0]
    gsc = estatisticas.loc[estatisticas["Time"]==timeHome, "gols_sofridos_casa"].iloc[0]
    gff = estatisticas.loc[estatisticas["Time"]==timeAway, "gols_feitos_fora"].iloc[0]
    gsf = estatisticas.loc[estatisticas["Time"]==timeAway, "gols_sofridos_fora"].iloc[0]
    #Media de Gols Fora e Gols em Casa
    avgGH = (gfc*gsf)/2
    avgGA = (gff*gsc)/2
    erroCasa = golsHome - avgGH
    erroFora = golsAway - avgGA
    
    linha['gols_feitos_casa'] = gfc
    linha['gols_feitos_fora'] = gff
    linha['gols_sofridos_casa'] = gsc
    linha['gols_sofridos_fora'] = gsf
    linha['media_gols_casa'] = avgGH
    linha['media_gols_fora'] = avgGA
    linha['erro_casa'] = erroCasa
    linha['erro_fora'] = erroFora
    return linha

tabela_erro = tabela_erro.apply(criar_tabela_erro, axis=1)

erroMandante = tabela_erro['erro_casa'].mean()
erroVisitante = tabela_erro['erro_fora'].mean()

def calculo_pontos(linha):
    time_casa = linha["Casa"]
    time_fora = linha["Fora"]
    
    lambda_casa = (estatisticas.loc[estatisticas['Time']==time_casa, "gols_feitos_casa"].iloc[0]*estatisticas.loc[estatisticas['Time']==time_fora, "gols_sofridos_fora"].iloc[0]) + erroMandante 
    lambda_fora = (estatisticas.loc[estatisticas['Time']==time_fora, "gols_feitos_fora"].iloc[0]*estatisticas.loc[estatisticas['Time']==time_casa, "gols_sofridos_casa"].iloc[0]) + erroVisitante
    
    
    pv_casa = 0
    p_empate = 0
    pv_fora = 0
    
    for i in range (11):
        for j in range(11):
            prob = poisson.pmf(i, lambda_casa)*poisson.pmf(j, lambda_fora)
            if i==j:
                p_empate += prob
            elif i>j:
                pv_casa += prob
            elif i<j:
                pv_fora += prob
    
    
    ve_casa = pv_casa*3 + p_empate
    ve_fora = pv_fora*3 + p_empate
    linha["pts_casa"] = ve_casa
    linha["pts_fora"] = ve_fora
    return linha
            
        
jogos_faltantes = jogos_faltantes.apply(calculo_pontos, axis=1)
pontos_casa = jogos_faltantes.groupby("Casa").sum(numeric_only=True)[["pts_casa"]]
pontos_fora = jogos_faltantes.groupby("Fora").sum(numeric_only=True)[["pts_fora"]]

def ajuste_classficacao(linha):
    for nome in nomeTimes:
        if nome in linha["Equipe"]:
            return nome
        
classificacao["time"] = classificacao.apply(ajuste_classficacao, axis=1)
        
classificacao_atualizada = classificacao[["time","Pts"]]


def atualizar_pts(linha):
    time = linha['time']
    pontos = int(linha["Pts"]) + float(pontos_casa.loc[time, "pts_casa"]) + float(pontos_fora.loc[time, "pts_fora"])
    return pontos
    
classificacao_atualizada["Pts"] = classificacao_atualizada.apply(atualizar_pts, axis=1)    
classificacao_atualizada = classificacao_atualizada.sort_values(by="Pts", ascending=False)  
classificacao_atualizada = classificacao_atualizada.reset_index(drop=True) 
classificacao_atualizada.index = classificacao_atualizada.index + 1    
    
    
#Calculo da probabilidade de um resultado     

# homeTeam = "Borussia Mönchengladbach"
# awayTeam = "Borussia Dortmund"

# lam_fora_solo = estatisticas.loc[estatisticas["Time"]==awayTeam, "gols_feitos_fora"].iloc[0]*estatisticas.loc[estatisticas["Time"]==homeTeam, "gols_sofridos_casa"].iloc[0] + erroMandante
# lam_casa_solo = estatisticas.loc[estatisticas["Time"]==homeTeam, "gols_feitos_casa"].iloc[0]*estatisticas.loc[estatisticas["Time"]==awayTeam, "gols_sofridos_fora"].iloc[0] + erroVisitante

# prob_df = pd.DataFrame()
   
# for m in range (6):
#     for n in range(6):
#         prob = (poisson.pmf(m, lam_casa_solo)*poisson.pmf(n, lam_fora_solo))*100
#         numero_gols = m + n
#         saldo_gols = m - n
#         golsHome = str(m)
#         golsAway = str(n)
#         result = golsHome + "-" + golsAway 
#         row = {"prob": prob, "resultado": result, "numero_gols": numero_gols, "saldo": saldo_gols}        
#         prob_df = prob_df._append(row, ignore_index=True)
        
           
# prob_df.sort_values(by='prob', ascending=False)
# prob_df.groupby("numero_gols").sum(numeric_only=True)["prob"]
# prob_df.groupby(prob_df["saldo"]).sum(numeric_only=True)["prob"]
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    



























