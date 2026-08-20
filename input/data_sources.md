# Fontes dos dados de entrada do PolicySpace3

Compilação a partir do que já está documentado em textos anteriores da plataforma,
do repositório de dados (`input/`) e do código que os carrega. Serve a qualquer
texto que precise declarar a procedência dos dados; foi escrita originalmente
para responder ao pedido dos pareceristas do Radar sobre as fontes do Quadro 1
(`text/Radar_plataforma/`). **Manter atualizada sempre que um dado de `input/`
mudar.**

Referências usadas nesta compilação:

- **PS1** — Furtado, B. A. *PolicySpace: agent-based modeling*. Brasília: Ipea, 2018
  (edição em inglês, seção 3.6 "ODD: input data", p. 44-46).
- **PS2** — Furtado, B. A. PolicySpace2: modeling markets and endogenous public
  policies. *JASSS*, v. 25, n. 1, 2022 (Apêndice, Tabela 4 "Table of input data
  entered in the model").
- **Emissões** — Rocha Lima, G. L.; Furtado, B. A.; Lopes, O. F. Innovation or
  contraction? *Energy Policy*, v. 211, 115095, 2026 (seção de dados).
- **Densidade** — Furtado et al. (em revisão), o artigo de densidade (`papers/density_housing_inequality/`)
  (Tabela de validação habitacional e descrição do mercado de crédito).
- **Repositório** — `input/Demografia/LEIA-ME.txt` e o processamento do Censo em
  <https://github.com/BAFurtado/censo2010>. Este arquivo incorpora e substitui o antigo
  `input/DataDescription.md`.

---

## 1. Nota do Quadro 1 — siglas por extenso

Texto sugerido para a nota (substituindo os `xxxxx`):

> Obs.: ACP – Área de Concentração Urbana; AP – área de ponderação;
> CSV – *comma-separated values* (formato de arquivo de texto com valores
> separados por vírgula); FGTS – Fundo de Garantia do Tempo de Serviço;
> FPM – Fundo de Participação dos Municípios; IDHM – Índice de Desenvolvimento
> Humano Municipal; MCMV – Programa Minha Casa, Minha Vida; PIB – produto interno
> bruto; QLI – *quality of life index* (índice de qualidade de vida calculado
> endogenamente pelo modelo, tendo o IDHM como referência de comparação);
> Rais – Relação Anual de Informações Sociais; SBPE – Sistema Brasileiro de
> Poupança e Empréstimo; Selic – Sistema Especial de Liquidação e de Custódia
> (taxa básica de juros).

Se as fontes forem incorporadas à nota, acrescentar também: Abecip – Associação
Brasileira das Entidades de Crédito Imobiliário e Poupança; BNDES – Banco
Nacional de Desenvolvimento Econômico e Social; CBIC – Câmara Brasileira da
Indústria da Construção; FJP – Fundação João Pinheiro;
IBGE – Instituto Brasileiro de Geografia e Estatística; Ibama – Instituto
Brasileiro do Meio Ambiente e dos Recursos Naturais Renováveis; MIP – matriz de
insumo-produto; PNUD – Programa das Nações Unidas para o Desenvolvimento;
RAPP – Relatório de Atividades Potencialmente Poluidoras e Utilizadoras de
Recursos Ambientais; STN – Secretaria do Tesouro Nacional.

---

## 2. Fontes das entradas empíricas (linha "Entradas empíricas" do Quadro 1)

Sugestão: transformar em um quadro próprio (Quadro 1A ou apêndice), já que a
lista completa não cabe na nota de rodapé.

| # | Entrada empírica | Fonte | Período | Arquivo no repositório | Documentação |
|---|---|---|---|---|---|
| 1 | População por sexo e idade, por área de ponderação e município | IBGE, Censo Demográfico 2000 e 2010 (Sidra, tabelas 1378 e 202) | 2000, 2010 | `input/num_people_age_gender_AP_*.csv`, `input/pop_{men,women}_*.csv` | PS2 Tab. 4 |
| 2 | Qualificação (anos de estudo) por área de ponderação, 5 níveis | IBGE, Censo Demográfico (Sidra, tabela 1554, <https://sidra.ibge.gov.br/tabela/1554>) | 2000, 2010 | `input/qualification_APs_*.csv` | PS2 Tab. 4 |
| 3 | Número médio de moradores por família, por área de ponderação | IBGE, Censo Demográfico 2010 | 2010 | `input/average_num_members_families_2010.csv` | PS2 Tab. 4 |
| 4 | Distribuição de qualidade dos domicílios particulares permanentes (DPP0–DPP5) por AP | IBGE, Censo Demográfico 2010 | 2010 | `input/dpp_2010_quali.csv` | processamento em `censo2010` |
| 5 | Proporção da população urbana por município | IBGE, Censos 2000, 2010 e 2022 (Sidra, tabela 202, <https://sidra.ibge.gov.br/tabela/202> — "População residente, por sexo e situação do domicílio") | 2000, 2010, 2022 | `input/Demografia/3_Percent_Urban/` | `LEIA-ME.txt` |
| 6 | Número de firmas por área de ponderação | Rais — Ministério do Trabalho e Emprego | 2000 e 2010 (cortes t0/t1) | `input/firms_by_APs{2000,2010}_t*_full.csv` | PS1 p. 45; PS2 Tab. 4 |
| 7 | Participação setorial do emprego por ACP (12 setores) | Rais — Ministério do Trabalho e Emprego | 2010 | `input/CONCURBs_SECTOR.csv` | PS2; paper de emissões |
| 8 | Vínculos ativos do setor Governo por município | Rais — Ministério do Trabalho e Emprego (série estabilizada a partir de 2020) | 2010–2045 | `input/qtde_vinc_gov_rais_stable_from_2020_onwards.csv` | — |
| 9 | Tábuas de fecundidade por UF e idade quinquenal (10–50 anos) | IBGE, Projeção da População do Brasil e Unidades da Federação | 2000–2070 | `input/Demografia/1_Fertility/` | PS1 p. 44; `LEIA-ME.txt` |
| 10 | Tábuas de mortalidade por UF, sexo e idade quinquenal (0–90+) | IBGE, mesma projeção; extrapolação acima de 90 anos conforme Castro (2015)/MPS | 2000–2070 | `input/Demografia/2_Mortality/` | PS1 p. 44 |
| 11 | Estimativas de população total por município | IBGE, estimativas populacionais enviadas ao TCU; projeções Cedeplar/UFMG na série estendida | 2001–2024 (TCU); 2000–2050 (Cedeplar) | `input/Demografia/4_Pop_Estimatives_Munic/` | `LEIA-ME.txt`; PS2 Tab. 4 |
| 12 | Domicílios totais e média de moradores por município | projeções de domicílios elaboradas pela equipe Ipea/PlanHab a partir da base demográfica acima | 2010–2040 | `input/Demografia/4_Pop_Estimatives_Munic/` | cálculo próprio |
| 13 | Idade ao casamento por sexo | IBGE | — | `input/marriage_age_{men,women}.csv` | — |
| 14 | Malhas territoriais (municípios, áreas de ponderação, manchas urbanas) | IBGE, malhas municipais digitais e shapefiles do Censo | 2010 / 2014 | `input/shapes/` | PS1 p. 45; PS2 Tab. 4 |
| 15 | Delimitação das Áreas de Concentração Urbana e composição municipal | IBGE. *Arranjos populacionais e concentrações urbanas do Brasil*. Rio de Janeiro, 2015 (atualizada com o Censo 2022) | 2015 / 2022 | `input/CONCURBs_BR.csv`, `input/CONCURBs_MUN_CODES.csv`, `input/ACPs_MUN_CODES.csv` | PS1 p. 45; PS2 |
| 16 | Matrizes insumo-produto regionalizadas por ACP (12 setores) | IBGE. *Matriz de Insumo-Produto 2015*, v. 62. Rio de Janeiro, 2018; regionalização conforme Miller e Blair (2022) | 2015 | `input/technical_matrices/*_matrix_io.json`, `input/technical_matrix*.csv` | paper de emissões; `text_density` |
| 17 | Demanda final por setor (exportações, consumo das famílias, FBCF, consumo do governo) | IBGE, MIP 2015 / Contas Nacionais | 2015 | `input/final_demand.csv`, `input/final_demand/*_final_demand.json` | idem |
| 18 | Matriz origem-destino de tempos de deslocamento (por área de ponderação) | BNDES — matriz de tempos de viagem produzida no âmbito do projeto de investimentos em transporte; hoje disponível apenas para o Distrito Federal | — | `input/bndes/travel_times_areapond_DF.parquet` | — |
| 19 | Séries de juros: Selic observada e taxa de financiamento imobiliário | Banco Central do Brasil, Sistema Gerenciador de Séries Temporais (SGS) — em PS2, série 25497 para o juro real de financiamento imobiliário | 2000–2050 (projetada após o último dado observado) | `input/interest_{real,media,alta,baixa,fixed}.csv` | PS2 Tab. 4; `text_density` |
| 20 | Juros regulados de crédito habitacional (SBPE/SFH e FGTS), por cenário | Banco Central do Brasil (série mensal SFH, FGTS, livre, comercial e *home equity*, 2014–2025; cenários fixados a partir da taxa média 2022–2025) | 2000–2050 | `input/planhab_funds/interest_housing_*.csv` | planilha `auxiliary/housing_data/inputs/Numeros_Bernardo.xlsx`, aba "juros" ("Fonte: Banco Central do Brasil") |
| 21 | Percentuais de recursos de FGTS e SBPE por município e ano, por cenário orçamentário | (a) histórico nacional de financiamento habitacional da Abecip; (b) cenários tendencial/otimista/pessimista de arrecadação, saques e ativos do FGTS e de estoque do SBPE, em reais de 2025; (c) pesos municipais calculados a partir dos contratos de empréstimo do FGTS por município e ano de assinatura (Caixa Econômica Federal); (d) PIB municipal do IBGE, para converter valores em proporção do PIB | 2002/2009–2050 | `input/planhab_funds/fgts_sbpe_pct_*.csv` | scripts `auxiliary/housing_data/PlanHab_data.py` e `pesos_fgts.py` |
| 22 | Repasses do FPM por município | Secretaria do Tesouro Nacional (STN) | 2000–2024 | `input/fpm/{UF}.csv` | PS1 p. 45 |
| 23 | Arrecadação tributária municipal (usada na validação da primeira geração) | STN, Siconfi/Finbra (<https://siconfi.tesouro.gov.br>) | — | — | PS1 p. 45 |
| 24 | Emissões setoriais e intensidade de emissão | Alvarenga Junior, M. *Towards a structural carbonization of the Brazilian economy*. Tese (doutorado), IE/UFRJ, 2024 (<https://www.ie.ufrj.br/images/IE/PPGE/teses/2024/Marcio%20Alverenga%20Junior%20-%20PhD%20Dissertation%20-%20TOWARDS%20A%20STRUCTURAL%20CARBONIZATION%20OF%20THE%20BRAZILIAN%20ECONOMY%20(26.02).pdf>) | — | `input/emissions_sectors.csv` | — |
| 25 | IDHM (referência para o QLI) | Atlas do Desenvolvimento Humano no Brasil — PNUD, Ipea e FJP (<http://atlasbrasil.org.br/2013>) | 2000 e 2010 | `input/idhm_2000_2010.csv` | PS1 p. 45; PS2 Tab. 4 |

Sobre o item 24: a intensidade de emissão setorial usada no artigo de emissões
(Rocha Lima, Furtado e Lopes, 2026) é construída a partir do RAPP do Ibama (2010)
combinado à massa salarial da Rais. Se a nota do Quadro 1 mencionar as duas
gerações do módulo, vale citar ambas as origens.

---

## 3. Dados dos experimentos de política habitacional (bloco de 16 parâmetros)

O Quadro 1 menciona um bloco à parte de parâmetros que configura experimentos de
política. Esses experimentos também são calibrados com dados observados, e vale
registrar as fontes caso o parecerista estenda a pergunta:

| Componente | Fonte | Onde está |
|---|---|---|
| Volume de recursos de FGTS e SBPE por município e cenário | Abecip (histórico), cenários de FGTS/SBPE em reais de 2025, contratos do FGTS por município, PIB municipal do IBGE | itens 20 e 21 da tabela acima |
| Faixa de elegibilidade do programa de melhorias (percentil 38 da renda; domicílios com índice de qualidade 0,5) | classificação empírica de inadequação habitacional no Brasil (Balbim, 2024, Ipea) | `text_density`, seção de desenho experimental |
| Montante de investimento em melhorias habitacionais | necessidade estimada de R$ 273,6 bilhões (valores de 2024) para sanar inadequações edilícias no período 2015–2040, distribuída entre municípios pela proporção de inadequações edilícias de cada um; convertida em proporção do PIB pelo PIB nacional do IBGE (março/2025, R$ 11,74 trilhões) — resulta em 0,24% do PIB anual, ou 0,0016 ao ano ao longo de 15 anos | `text/other/memoria_calculo_investimentos_melhorias.txt`; Balbim (2024), Ipea |
| Faixa 1 do MCMV como referência de focalização | regras do programa Minha Casa, Minha Vida; avaliação CMAP/CMAS (2021) | avaliação CMAP/CMAS do MCMV (documento de trabalho) |

---

## 4. Fontes das faixas empíricas usadas na calibragem/validação

A linha "Calibragem e validação" do Quadro 1 também usa dados externos. Estas
fontes já estão consolidadas na tabela de validação do texto de densidade e
podem ser citadas diretamente:

| Indicador de referência | Faixa empírica | Fonte |
|---|---|---|
| Estoque de imóveis / PIB | 1,5–2,5 | Banco Central do Brasil, *Relatório de Estabilidade Financeira* (2019) |
| Razão preço/salário mensal | 60–150 | Fundação João Pinheiro, *Déficit Habitacional no Brasil* (2022) |
| Razão preço/renda anual | 6,0–12,0 | Banco Central do Brasil, *Relatório de Estabilidade Financeira* (2023) |
| Taxa de vacância | 0,08–0,13 | IBGE, Censos Demográficos 2010 e 2022 |
| Consumo / PIB | 0,55–0,65 | IBGE, Contas Nacionais (média 2010–2024) |
| Produção habitacional por mil habitantes | 2,0–4,0 | Câmara Brasileira da Indústria da Construção (CBIC, 2019) |
| Estoque habitacional / renda permanente | 5,0–8,0 | estimativa dos autores a partir de pesquisas domiciliares nacionais |
| Séries usadas na validação temporal | — | Ipeadata: IPCA e PIB mensal a preços correntes, deflacionado pelo IPCA (série sem ajuste sazonal) |

---

## 5. Texto pronto para a nota do Quadro 1

Versão curta, para caber na nota:

> Fonte: elaboração dos autores. Entradas empíricas provenientes de: IBGE
> (Censos Demográficos 2000, 2010 e 2022; projeções de fecundidade e mortalidade
> por unidade da federação; estimativas municipais de população; malhas
> territoriais; delimitação das Áreas de Concentração Urbana; Matriz de
> Insumo-Produto 2015; PIB dos municípios); Rais/Ministério do Trabalho e Emprego
> (firmas por área de ponderação, composição setorial do emprego e vínculos do
> setor Governo); Secretaria do Tesouro Nacional (repasses do FPM); Banco Central
> do Brasil (séries de juros e taxas reguladas de crédito habitacional); Abecip e
> Caixa Econômica Federal (financiamento habitacional e contratos do FGTS por
> município); BNDES (matriz origem-destino de tempos de deslocamento);
> Alvarenga Junior (2024) (emissões setoriais); e Atlas do Desenvolvimento Humano
> no Brasil — PNUD, Ipea e FJP (IDHM 2000–2010). As projeções municipais de
> domicílios são cálculo próprio da equipe. O detalhamento por variável, com
> período e arquivo correspondente, consta do Quadro 1A / Apêndice.

Versão para apêndice: usar a tabela da seção 2 acima.