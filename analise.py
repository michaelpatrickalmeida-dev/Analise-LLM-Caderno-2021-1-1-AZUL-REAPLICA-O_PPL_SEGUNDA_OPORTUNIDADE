# -*- coding: utf-8 -*-
import os
import re
import json
import glob
import time
import requests
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("Defina GEMINI_API_KEY no .env")

# --- CONFIGURAÇÕES DA PROVA DO MICHAEL ---
MODELO = "gemini-3.5-flash-lite"
PASTA_TEXTOS  = "textos_extraidos"
PASTA_IMAGENS = "imagens"
PASTA_SAIDA   = "resultados"
ARQ_ESTRUTURA = "estrutura.json"
ARQ_MATRIZ    = "matriz_referencia_enem.json"
ARQ_GABARITO  = "gabarito.txt"

IDENTIFICACAO_CADERNO = {
    "ano": 2021,
    "dia": 1,
    "numero": 1,
    "cor": "AZUL",
    "aplicacao": "REAPLICAÇÃO_PPL_SEGUNDA_OPORTUNIDADE",
}

ESTUDANTE = {
    "nome_completo": "Michael Patrick Almeida",
    "colegio": "CCM Darcy José Costa",
    "turma": "3B",
}

TAMANHO_LOTE = 10
ESPERA_ENTRE_LOTES = 65
os.makedirs(PASTA_SAIDA, exist_ok=True)

# --- FUNÇÕES DE APOIO ---
def carregar_gabarito(caminho):
    gabarito = {}
    if not os.path.exists(caminho): return gabarito
    padrao = re.compile(r"^\s*(\d+)\s*[-\.\)]?\s*([A-Ea-e])(?:\s*\((\w+)\))?\s*$")
    with open(caminho, encoding="utf-8") as f:
        for l in f:
            l = l.strip()
            if not l or l.startswith("#"): continue
            m = padrao.match(l)
            if m:
                num, letra, idioma = int(m.group(1)), m.group(2).upper(), m.group(3)
                if idioma: gabarito[(num, idioma.lower())] = letra
                else: gabarito[num] = letra
    return gabarito

def carregar_json(c):
    with open(c, encoding="utf-8") as f: return json.load(f)

def parse_nome_arquivo(nome):
    stem = Path(nome).stem.lower()
    info = {"numero": None, "ingles": False, "espanhol": False}
    m = re.match(r"questao-(ingles|espanhol)-(\d+)", stem)
    if m:
        info["numero"], info["ingles"], info["espanhol"] = int(m.group(2)), m.group(1)=="ingles", m.group(1)=="espanhol"
        return info
    m = re.match(r"questao-(\d+)", stem)
    if m: info["numero"] = int(m.group(1)); return info
    return info

# --- PROMPT ---
SUGESTION_PROMPT = """Você é um analista do ENEM. Classifique a questão segundo a Matriz.
Retorne APENAS o JSON no schema fornecido. Não invente itens fora da Matriz.
Atenção: NÃO use caracteres de barra invertida (como \\pi ou \\frac) no JSON. Descreva as fórmulas em texto puro."""

def montar_prompt(txt, matriz, estrutura, gab, meta):
    idioma = "português"
    if meta.get("ingles"): idioma = "inglês"
    elif meta.get("espanhol"): idioma = "espanhol"
    return f"""{SUGESTION_PROMPT}
MATRIZ: {json.dumps(matriz, ensure_ascii=False)}
SCHEMA: {json.dumps(estrutura, ensure_ascii=False)}
METADADOS: Nº {meta.get('numero')} | Idioma: {idioma} | Ano: {IDENTIFICACAO_CADERNO['ano']}
GABARITO: {gab if gab else "NÃO DISPONÍVEL"}
TEXTO: \"\"\"{txt}\"\"\"
Retorne o JSON."""

# --- CHAMADA DA API ---
def analisar_questao(txt, matriz, estrutura, gab, meta):
    prompt = montar_prompt(txt, matriz, estrutura, gab, meta)
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODELO}:generateContent?key={GEMINI_API_KEY}"
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.0,
            "responseMimeType": "application/json",
        }
    }

    for t in range(5):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=60)
            if resp.status_code == 200:
                dados = resp.json()
                texto = dados['candidates'][0]['content']['parts'][0]['text']
                texto = re.sub(r"^```(?:json)?|```$", "", texto, flags=re.MULTILINE).strip()
                texto = re.sub(r'\\(?![\\"/bfnrtu])', r'\\\\', texto)
                texto = texto.replace("\n", " ").replace("\r", " ")
                return json.loads(texto)
            elif resp.status_code == 429:
                print(f"    [COTA] Esperando 70s... ({t+1}/5)")
                time.sleep(70)
            elif resp.status_code == 503:
                print(f"    [SERVIDOR LOTADO] Esperando 15s... ({t+1}/5)")
                time.sleep(15)
            else:
                print(f"    [ERRO HTTP {resp.status_code}] {resp.text[:150]}")
                time.sleep(5)
        except Exception as e:
            print(f"    [ERRO DE JSON] {t+1}/5: {e}")
            time.sleep(5)
    return None

# --- PROCESSAMENTO ---
def processar_questao(arq, matriz, estrutura, gabarito):
    with open(arq, encoding="utf-8") as f: txt = f.read().strip()
    if not txt: return None
    meta = parse_nome_arquivo(arq)
    num = meta["numero"]
    gab = None
    if num:
        if meta.get("ingles") and (num, "ingles") in gabarito: gab = gabarito[(num, "ingles")]
        elif meta.get("espanhol") and (num, "espanhol") in gabarito: gab = gabarito[(num, "espanhol")]
        else: gab = gabarito.get(num)
    print(f"\n>>> {arq} (nº {num}) gab={gab}")
    res = analisar_questao(txt, matriz, estrutura, gab, meta)
    if not res: return None
    
    q = res.setdefault("questao", {})
    q.setdefault("identificacao_caderno", {}).update(IDENTIFICACAO_CADERNO)
    iq = q.setdefault("identificacao_questao", {})
    iq.update({"numero": num, "ingles": meta.get("ingles"), "espanhol": meta.get("espanhol")})
    cq = q.setdefault("conteudo_questao", {})
    cq["caminho_texto"] = arq.replace("\\", "/")
    if gab: cq["gabarito"] = gab
    res["estudante"] = ESTUDANTE
    
    nome_base = Path(arq).stem
    with open(os.path.join(PASTA_SAIDA, f"{nome_base}.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    return res

def main():
    print("ANÁLISE LLM — ENEM 2021-1-1 AZUL (Reaplicação PPL)")
    matriz = carregar_json(ARQ_MATRIZ)
    estrutura = carregar_json(ARQ_ESTRUTURA)
    gabarito = carregar_gabarito(ARQ_GABARITO)
    print(f"[INFO] Gabarito: {len(gabarito)} itens.")
    
    arquivos = sorted(glob.glob(os.path.join(PASTA_TEXTOS, "*.txt")))
    arquivos = [a for a in arquivos if Path(a).stem.lower() != "readme"]
    total = len(arquivos)
    total_lotes = (total + TAMANHO_LOTE - 1) // TAMANHO_LOTE
    print(f"[INFO] {total} arquivos / {total_lotes} lotes.\n")
    
    consolidado = []
    for idx in range(total_lotes):
        inicio = idx * TAMANHO_LOTE
        lote = arquivos[inicio:inicio+TAMANHO_LOTE]
        print(f"\n📦 LOTE {idx+1}/{total_lotes}")
        for arq in tqdm(lote, desc=f"Lote {idx+1}"):
            r = processar_questao(arq, matriz, estrutura, gabarito)
            if r: consolidado.append(r)
        if idx < total_lotes - 1:
            print(f"\n⏳ Esperando {ESPERA_ENTRE_LOTES}s...")
            time.sleep(ESPERA_ENTRE_LOTES)
    
    with open(os.path.join(PASTA_SAIDA, "analise_consolidada.json"), "w", encoding="utf-8") as f:
        json.dump(consolidado, f, ensure_ascii=False, indent=2)
    print(f"\n✅ Total: {len(consolidado)}/{total}")

if __name__ == "__main__":
    main()