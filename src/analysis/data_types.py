"""Data types for experiment results analysis."""

from dataclasses import dataclass
from typing import Optional
import statistics


@dataclass
class GameRun:
    """Representa um único jogo (uma linha do CSV).
    
    Attributes:
        target_id: ID do nó objetivo
        target_label: Nome da cidade objetivo
        run_index: Índice da execução (para múltiplas runs da mesma cidade)
        turns: Número de turnos jogados
        h_start: Entropia inicial
        h_end: Entropia final
        total_info_gain: Ganho total de informação
        avg_info_gain_per_turn: Ganho de informação médio por turno
        win: Se o seeker venceu
        compliance_rate: Taxa de compliance do oracle
        conversation_path: Caminho relativo para conversas salvas
        seeker_total_tokens: Total de tokens do Seeker
        seeker_reasoning_tokens: Tokens de reasoning (None se não houver)
        seeker_final_tokens: Tokens da resposta final
    """
    target_id: str
    target_label: str
    run_index: int
    turns: int
    h_start: float
    h_end: float
    total_info_gain: float
    avg_info_gain_per_turn: float
    win: bool
    compliance_rate: float
    conversation_path: Optional[str] = None
    seeker_total_tokens: int = 0
    seeker_reasoning_tokens: Optional[int] = None
    seeker_final_tokens: int = 0


@dataclass
class CityStats:
    """Estatísticas agregadas para uma cidade específica.
    
    Attributes:
        city_id: ID da cidade
        city_label: Nome da cidade
        runs: Lista de jogos dessa cidade
    """
    city_id: str
    city_label: str
    runs: list[GameRun]
    
    @property
    def num_runs(self) -> int:
        """Número total de runs desta cidade."""
        return len(self.runs)
    
    @property
    def mean_info_gain(self) -> float:
        """Ganho de informação médio desta cidade."""
        if not self.runs:
            return 0.0
        return sum(r.total_info_gain for r in self.runs) / self.num_runs
    
    @property
    def var_info_gain(self) -> float:
        """Variância do ganho de informação desta cidade (populacional)."""
        if not self.runs or self.num_runs == 1:
            return 0.0
        values = [r.total_info_gain for r in self.runs]
        return statistics.pvariance(values)
    
    @property
    def std_info_gain(self) -> float:
        """Desvio padrão do ganho de informação (populacional)."""
        if not self.runs or self.num_runs == 1:
            return 0.0
        values = [r.total_info_gain for r in self.runs]
        return statistics.pstdev(values)
    
    @property
    def win_rate(self) -> float:
        """Taxa de vitória nesta cidade."""
        if not self.runs:
            return 0.0
        return sum(1 for r in self.runs if r.win) / self.num_runs
    
    @property
    def mean_turns(self) -> float:
        """Média de turnos nesta cidade."""
        if not self.runs:
            return 0.0
        return sum(r.turns for r in self.runs) / self.num_runs
    
    @property
    def mean_h_start(self) -> float:
        """Média da entropia inicial (H no início do jogo) nesta cidade."""
        if not self.runs:
            return 0.0
        return sum(r.h_start for r in self.runs) / self.num_runs
    
    @property
    def std_turns(self) -> float:
        """Desvio padrão de turnos (populacional)."""
        if not self.runs or self.num_runs == 1:
            return 0.0
        values = [r.turns for r in self.runs]
        return statistics.pstdev(values)
    
    @property
    def mean_avg_info_gain_per_turn(self) -> float:
        """Média do ganho de informação médio por turno."""
        if not self.runs:
            return 0.0
        return sum(r.avg_info_gain_per_turn for r in self.runs) / self.num_runs
    
    @property
    def std_avg_info_gain_per_turn(self) -> float:
        """Desvio padrão do ganho de informação médio por turno (populacional)."""
        if not self.runs or self.num_runs == 1:
            return 0.0
        values = [r.avg_info_gain_per_turn for r in self.runs]
        return statistics.pstdev(values)
    
    @property
    def mean_seeker_tokens(self) -> float:
        """Média de tokens do Seeker nesta cidade."""
        if not self.runs:
            return 0.0
        return sum(r.seeker_total_tokens for r in self.runs) / self.num_runs
    
    @property
    def mean_seeker_reasoning_tokens(self) -> Optional[float]:
        """Média de tokens de reasoning do Seeker (None se nenhum run tiver reasoning)."""
        runs_with_reasoning = [r for r in self.runs if r.seeker_reasoning_tokens is not None]
        if not runs_with_reasoning:
            return None
        total = sum(r.seeker_reasoning_tokens for r in runs_with_reasoning if r.seeker_reasoning_tokens is not None)
        return total / len(runs_with_reasoning)
    
    @property
    def mean_seeker_final_tokens(self) -> float:
        """Média de tokens da resposta final do Seeker."""
        if not self.runs:
            return 0.0
        return sum(r.seeker_final_tokens for r in self.runs) / self.num_runs


@dataclass
class ExperimentResults:
    """Resultados completos de um experimento.
    
    Attributes:
        experiment_name: Nome do experimento
        seeker_model: Modelo usado no Seeker
        oracle_model: Modelo usado no Oracle
        pruner_model: Modelo usado no Pruner
        observability: Modo de observabilidade
        max_turns: Número máximo de turnos
        cities: Dicionário {city_id: CityStats}
    """
    experiment_name: str
    seeker_model: str
    oracle_model: str
    pruner_model: str
    observability: str
    max_turns: int
    cities: dict[str, CityStats]
    
    @property
    def total_runs(self) -> int:
        """Número total de runs no experimento."""
        return sum(city.num_runs for city in self.cities.values())
    
    @property
    def mean_info_gain(self) -> float:
        """Ganho de informação médio global (média das médias por cidade)."""
        if not self.cities:
            return 0.0
        return sum(city.mean_info_gain for city in self.cities.values()) / len(self.cities)
    
    @property
    def global_win_rate(self) -> float:
        """Taxa de vitória global."""
        if self.total_runs == 0:
            return 0.0
        total_wins = sum(sum(1 for r in city.runs if r.win) for city in self.cities.values())
        return total_wins / self.total_runs
    
    @property
    def mean_turns(self) -> float:
        """Média global de turnos."""
        if self.total_runs == 0:
            return 0.0
        total_turns = sum(sum(r.turns for r in city.runs) for city in self.cities.values())
        return total_turns / self.total_runs
    
    @property
    def mean_compliance(self) -> float:
        """Taxa de compliance média global."""
        if self.total_runs == 0:
            return 0.0
        total_compliance = sum(sum(r.compliance_rate for r in city.runs) for city in self.cities.values())
        return total_compliance / self.total_runs
    
    @property
    def mean_avg_info_gain_per_turn(self) -> float:
        """Ganho de informação médio por turno global."""
        if self.total_runs == 0:
            return 0.0
        total_avg_gi = sum(sum(r.avg_info_gain_per_turn for r in city.runs) for city in self.cities.values())
        return total_avg_gi / self.total_runs
    
    @property
    def mean_seeker_tokens(self) -> float:
        """Média global de tokens do Seeker."""
        if self.total_runs == 0:
            return 0.0
        total_tokens = sum(sum(r.seeker_total_tokens for r in city.runs) for city in self.cities.values())
        return total_tokens / self.total_runs
    
    @property
    def mean_seeker_reasoning_tokens(self) -> Optional[float]:
        """Média global de tokens de reasoning (None se nenhum run tiver reasoning)."""
        all_runs = [r for city in self.cities.values() for r in city.runs]
        runs_with_reasoning = [r for r in all_runs if r.seeker_reasoning_tokens is not None]
        if not runs_with_reasoning:
            return None
        total = sum(r.seeker_reasoning_tokens for r in runs_with_reasoning if r.seeker_reasoning_tokens is not None)
        return total / len(runs_with_reasoning)
    
    @property
    def mean_seeker_final_tokens(self) -> float:
        """Média global de tokens da resposta final do Seeker."""
        if self.total_runs == 0:
            return 0.0
        total_tokens = sum(sum(r.seeker_final_tokens for r in city.runs) for city in self.cities.values())
        return total_tokens / self.total_runs
    
    def _get_all_runs(self) -> list[GameRun]:
        """Retorna lista de todos os runs de todas as cidades."""
        return [r for city in self.cities.values() for r in city.runs]
    
    @property
    def std_info_gain(self) -> float:
        """Desvio padrão global do ganho de informação (calculado sobre médias por cidade).
        
        Usa abordagem hierárquica: calcula std sobre as médias de cada cidade,
        tratando cada cidade como uma unidade de amostragem independente.
        Isso mede a capacidade de generalização através do domínio geográfico.
        Usa variância amostral (N-1) para inferência estatística.
        """
        if len(self.cities) <= 1:
            return 0.0
        city_means = [city.mean_info_gain for city in self.cities.values()]
        return statistics.stdev(city_means) if len(city_means) > 1 else 0.0
    
    @property
    def std_avg_info_gain_per_turn(self) -> float:
        """Desvio padrão global do ganho de informação médio por turno (calculado sobre médias por cidade).
        
        Usa abordagem hierárquica: calcula std sobre as médias de cada cidade.
        Usa variância amostral (N-1) para inferência estatística.
        """
        if len(self.cities) <= 1:
            return 0.0
        city_means = [city.mean_avg_info_gain_per_turn for city in self.cities.values()]
        return statistics.stdev(city_means) if len(city_means) > 1 else 0.0
    
    @property
    def std_turns(self) -> float:
        """Desvio padrão global de turnos (calculado sobre médias por cidade).
        
        Usa abordagem hierárquica: calcula std sobre as médias de cada cidade.
        Usa variância amostral (N-1) para inferência estatística.
        """
        if len(self.cities) <= 1:
            return 0.0
        city_means = [city.mean_turns for city in self.cities.values()]
        return statistics.stdev(city_means) if len(city_means) > 1 else 0.0
    
    @property
    def std_seeker_tokens(self) -> float:
        """Desvio padrão global de tokens do Seeker (calculado sobre médias por cidade).
        
        Usa abordagem hierárquica: calcula std sobre as médias de cada cidade.
        Usa variância amostral (N-1) para inferência estatística.
        """
        if len(self.cities) <= 1:
            return 0.0
        city_means = [city.mean_seeker_tokens for city in self.cities.values()]
        return statistics.stdev(city_means) if len(city_means) > 1 else 0.0
    
    @property
    def std_seeker_reasoning_tokens(self) -> Optional[float]:
        """Desvio padrão global de tokens de reasoning (calculado sobre médias por cidade).
        
        Retorna None se nenhuma cidade tiver runs com reasoning.
        Usa abordagem hierárquica: calcula std sobre as médias de cada cidade.
        Usa variância amostral (N-1) para inferência estatística.
        """
        cities_with_reasoning = [
            city for city in self.cities.values() 
            if city.mean_seeker_reasoning_tokens is not None
        ]
        if len(cities_with_reasoning) <= 1:
            return None if not cities_with_reasoning else 0.0
        city_means = [
            city.mean_seeker_reasoning_tokens 
            for city in cities_with_reasoning 
            if city.mean_seeker_reasoning_tokens is not None
        ]
        if not city_means:
            return None
        return statistics.stdev(city_means) if len(city_means) > 1 else 0.0
    
    @property
    def std_seeker_final_tokens(self) -> float:
        """Desvio padrão global de tokens da resposta final do Seeker (calculado sobre médias por cidade).
        
        Usa abordagem hierárquica: calcula std sobre as médias de cada cidade.
        Usa variância amostral (N-1) para inferência estatística.
        """
        if len(self.cities) <= 1:
            return 0.0
        city_means = [city.mean_seeker_final_tokens for city in self.cities.values()]
        return statistics.stdev(city_means) if len(city_means) > 1 else 0.0
    
    @property
    def se_mean_info_gain(self) -> float:
        """Standard Error da média global de ganho de informação.
        
        Calculado usando std sobre médias por cidade, dividido por sqrt(n_cities).
        Isso reflete a incerteza ao generalizar para novas cidades.
        """
        if len(self.cities) <= 1:
            return 0.0
        return self.std_info_gain / (len(self.cities) ** 0.5)
    
    @property
    def se_mean_avg_info_gain_per_turn(self) -> float:
        """Standard Error da média global de ganho de informação por turno.
        
        Calculado usando std sobre médias por cidade, dividido por sqrt(n_cities).
        """
        if len(self.cities) <= 1:
            return 0.0
        return self.std_avg_info_gain_per_turn / (len(self.cities) ** 0.5)
    
    @property
    def se_mean_turns(self) -> float:
        """Standard Error da média global de turnos.
        
        Calculado usando std sobre médias por cidade, dividido por sqrt(n_cities).
        """
        if len(self.cities) <= 1:
            return 0.0
        return self.std_turns / (len(self.cities) ** 0.5)
    
    @property
    def mean_h_start(self) -> float:
        """Média global da entropia inicial (média sobre todas as runs)."""
        if self.total_runs == 0:
            return 0.0
        total = sum(sum(r.h_start for r in city.runs) for city in self.cities.values())
        return total / self.total_runs
    
    @property
    def std_h_start(self) -> float:
        """Desvio padrão da entropia inicial (sobre médias por cidade)."""
        if len(self.cities) <= 1:
            return 0.0
        city_means = [city.mean_h_start for city in self.cities.values()]
        return statistics.stdev(city_means) if len(city_means) > 1 else 0.0
    
    @property
    def se_mean_h_start(self) -> float:
        """Standard Error da média global da entropia inicial."""
        if len(self.cities) <= 1:
            return 0.0
        return self.std_h_start / (len(self.cities) ** 0.5)
    
    @property
    def se_mean_seeker_tokens(self) -> float:
        """Standard Error da média global de tokens do Seeker.
        
        Calculado usando std sobre médias por cidade, dividido por sqrt(n_cities).
        """
        if len(self.cities) <= 1:
            return 0.0
        return self.std_seeker_tokens / (len(self.cities) ** 0.5)
    
    @property
    def se_mean_seeker_reasoning_tokens(self) -> Optional[float]:
        """Standard Error da média global de tokens de reasoning (None se nenhuma cidade tiver reasoning).
        
        Calculado usando std sobre médias por cidade, dividido por sqrt(n_cities_com_reasoning).
        """
        cities_with_reasoning = [
            city for city in self.cities.values() 
            if city.mean_seeker_reasoning_tokens is not None
        ]
        if len(cities_with_reasoning) <= 1:
            return None if not cities_with_reasoning else 0.0
        std = self.std_seeker_reasoning_tokens
        if std is None:
            return None
        return std / (len(cities_with_reasoning) ** 0.5)
    
    @property
    def se_mean_seeker_final_tokens(self) -> float:
        """Standard Error da média global de tokens da resposta final do Seeker.
        
        Calculado usando std sobre médias por cidade, dividido por sqrt(n_cities).
        """
        if len(self.cities) <= 1:
            return 0.0
        return self.std_seeker_final_tokens / (len(self.cities) ** 0.5)
    
    @property
    def std_win_rate(self) -> float:
        """Desvio padrão global do win rate (calculado sobre win rates por cidade).
        
        Usa abordagem hierárquica: calcula std sobre os win rates de cada cidade.
        Usa variância amostral (N-1) para inferência estatística.
        """
        if len(self.cities) <= 1:
            return 0.0
        city_win_rates = [city.win_rate for city in self.cities.values()]
        return statistics.stdev(city_win_rates) if len(city_win_rates) > 1 else 0.0
    
    @property
    def se_win_rate(self) -> float:
        """Standard Error do win rate global.
        
        Calculado usando std sobre win rates por cidade, dividido por sqrt(n_cities).
        Isso reflete a incerteza ao generalizar para novas cidades.
        """
        if len(self.cities) <= 1:
            return 0.0
        return self.std_win_rate / (len(self.cities) ** 0.5)
    
    def get_city(self, city_id: str) -> Optional[CityStats]:
        """Retorna estatísticas de uma cidade específica."""
        return self.cities.get(city_id)
    
    def summary_dict(self) -> dict:
        """Retorna um dicionário com resumo completo do experimento."""
        return {
            "experiment_name": self.experiment_name,
            "models": {
                "seeker": self.seeker_model,
                "oracle": self.oracle_model,
                "pruner": self.pruner_model,
            },
            "config": {
                "observability": self.observability,
                "max_turns": self.max_turns,
            },
            "global_metrics": {
                "total_runs": self.total_runs,
                "total_cities": len(self.cities),
                "mean_info_gain": round(self.mean_info_gain, 4),
                "std_info_gain": round(self.std_info_gain, 4),
                "se_mean_info_gain": round(self.se_mean_info_gain, 4),
                "mean_avg_info_gain_per_turn": round(self.mean_avg_info_gain_per_turn, 4),
                "std_avg_info_gain_per_turn": round(self.std_avg_info_gain_per_turn, 4),
                "se_mean_avg_info_gain_per_turn": round(self.se_mean_avg_info_gain_per_turn, 4),
                "win_rate": round(self.global_win_rate, 4),
                "std_win_rate": round(self.std_win_rate, 4),
                "se_win_rate": round(self.se_win_rate, 4),
                "mean_turns": round(self.mean_turns, 2),
                "std_turns": round(self.std_turns, 2),
                "se_mean_turns": round(self.se_mean_turns, 2),
                "mean_h_start": round(self.mean_h_start, 4),
                "std_h_start": round(self.std_h_start, 4),
                "se_mean_h_start": round(self.se_mean_h_start, 4),
                "mean_compliance": round(self.mean_compliance, 4),
                "mean_seeker_tokens": round(self.mean_seeker_tokens, 0),
                "std_seeker_tokens": round(self.std_seeker_tokens, 0),
                "se_mean_seeker_tokens": round(self.se_mean_seeker_tokens, 0),
                "mean_seeker_reasoning_tokens": round(self.mean_seeker_reasoning_tokens, 0) if self.mean_seeker_reasoning_tokens is not None else None,
                "std_seeker_reasoning_tokens": round(self.std_seeker_reasoning_tokens, 0) if self.std_seeker_reasoning_tokens is not None else None,
                "se_mean_seeker_reasoning_tokens": round(self.se_mean_seeker_reasoning_tokens, 0) if self.se_mean_seeker_reasoning_tokens is not None else None,
                "mean_seeker_final_tokens": round(self.mean_seeker_final_tokens, 0),
                "std_seeker_final_tokens": round(self.std_seeker_final_tokens, 0),
                "se_mean_seeker_final_tokens": round(self.se_mean_seeker_final_tokens, 0),
            },
            "by_city": {
                city_id: {
                    "label": city.city_label,
                    "runs": city.num_runs,
                    "mean_info_gain": round(city.mean_info_gain, 4),
                    "std_info_gain": round(city.std_info_gain, 4),
                    "var_info_gain": round(city.var_info_gain, 4),
                    "mean_avg_info_gain_per_turn": round(city.mean_avg_info_gain_per_turn, 4),
                    "std_avg_info_gain_per_turn": round(city.std_avg_info_gain_per_turn, 4),
                    "win_rate": round(city.win_rate, 4),
                    "mean_turns": round(city.mean_turns, 2),
                    "std_turns": round(city.std_turns, 2),
                    "mean_seeker_tokens": round(city.mean_seeker_tokens, 0),
                    "mean_seeker_reasoning_tokens": round(city.mean_seeker_reasoning_tokens, 0) if city.mean_seeker_reasoning_tokens is not None else None,
                    "mean_seeker_final_tokens": round(city.mean_seeker_final_tokens, 0),
                }
                for city_id, city in sorted(self.cities.items())
            }
        }

