"""Motor de inferência ternária (−1, 0, +1)."""

from __future__ import annotations

import random
from dataclasses import dataclass


NEG = -1
NEU = 0
POS = 1

STATE_LABELS = {NEG: "INIBIÇÃO", NEU: "NEUTRO", POS: "ATIVAÇÃO"}


@dataclass
class InferenceResult:
    result: int
    confidence: float
    distribution: dict[str, int]


class TernaryEngine:
    """Classifica entradas em estados ternários com distribuição de confiança."""

    NEG_WORDS = ("não", "erro", "falha", "problema", "negativo", "ruim", "alerta", "crítico", "reiniciar")
    POS_WORDS = ("sim", "ok", "bom", "positivo", "sucesso", "ativo", "online", "pronto", "ótimo")
    NEU_WORDS = ("status", "diagnóstico", "diagnostico", "info", "analise", "análise", "consulta", "verificar")

    def __init__(self) -> None:
        self.distribution = {"neg": 33, "neu": 34, "pos": 33}

    def infer(self, text: str) -> InferenceResult:
        cmd = text.lower().strip()
        score = 0.0

        for w in self.NEG_WORDS:
            if w in cmd:
                score -= 1
        for w in self.POS_WORDS:
            if w in cmd:
                score += 1
        for w in self.NEU_WORDS:
            if w in cmd:
                score *= 0.5

        score += (random.random() - 0.5) * 0.6

        if score < -0.3:
            result, confidence = NEG, min(95.0, 60.0 + abs(score) * 20)
        elif score > 0.3:
            result, confidence = POS, min(95.0, 60.0 + abs(score) * 20)
        else:
            result, confidence = NEU, min(90.0, 55.0 + random.random() * 25)

        spread = confidence / 100
        total = 100
        dist = {
            "neg": round(spread * total) if result == NEG else round((1 - spread) * total * 0.4),
            "neu": round(spread * total) if result == NEU else round((1 - spread) * total * 0.3),
            "pos": round(spread * total) if result == POS else round((1 - spread) * total * 0.3),
        }
        dist["neu"] += total - sum(dist.values())
        self.distribution = dist

        return InferenceResult(result=result, confidence=round(confidence, 1), distribution=dist)

    def respond(self, text: str, inference: InferenceResult, telemetry: dict[str, str] | None = None) -> str:
        cmd = text.lower().strip()
        t = telemetry or {}

        if "status" in cmd:
            return (
                f"Status do sistema: ONLINE. Neurônios ternários: 4.096 ativos. "
                f"Convergência: {t.get('convergence', '98%')}. Núcleo: {t.get('temp', '37°C')}."
            )
        if "diagnóstico" in cmd or "diagnostico" in cmd:
            return (
                "Diagnóstico completo: 12 camadas neurais OK. "
                "Memória ternária: 98% livre. Latência média: 12ms. Nenhuma falha detectada."
            )
        if "reiniciar" in cmd:
            return "Reiniciando núcleo ternário... Sequência de boot concluída. Sistema restaurado."
        if "analise" in cmd or "análise" in cmd:
            d = inference.distribution
            return (
                f"Análise ternária: −1: {d['neg']}%, 0: {d['neu']}%, +1: {d['pos']}%. "
                f"Confiança: {inference.confidence}%."
            )

        responses = {
            NEG: [
                "Detectei anomalias no fluxo ternário. Recomendo verificação dos vetores de inibição.",
                "Estado −1 confirmado. Sistemas de segurança em alerta.",
                "Análise negativa concluída. Parâmetros fora do intervalo ideal.",
            ],
            NEU: [
                "Consulta processada. Estado neutro — nenhuma ação imediata necessária.",
                "Diagnóstico completo. Subsistemas dentro dos parâmetros normais.",
                "Informação registrada. Modelo ternário em equilíbrio.",
            ],
            POS: [
                "Confirmação positiva. Todos os indicadores operacionais.",
                "Estado +1 atingido. Otimização neural concluída com sucesso.",
                "Processamento concluído. Resposta favorável nos neurônios ternários.",
            ],
        }
        return random.choice(responses[inference.result])
