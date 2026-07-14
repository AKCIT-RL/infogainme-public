"""Domain configuration for different game datasets (geo, flat objects)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class DomainConfig:
    """Configuracao do dominio do jogo.

    Atributos:
        leaf_type: Tipo de no folha ("city" | "object").
        node_id_prefix: Prefixo dos IDs dos nos folha ("city:" | "object:").
        target_noun: Termo para o alvo nos prompts ("city" | "object").
        domain_description: Descricao do dominio para prompts (shared by all agents).
        seeker_pool_description: Optional extra context injected only into the
            Seeker system prompt. Leave empty for baseline behaviour (no prior).
    """

    leaf_type: str
    node_id_prefix: str
    target_noun: str
    domain_description: str
    seeker_pool_description: str = field(default="")


# Configuracoes pre-definidas

GEO_DOMAIN = DomainConfig(
    leaf_type="city",
    node_id_prefix="city:",
    target_noun="city",
    domain_description="geographic (cities, countries, regions)",
)

OBJECTS_DOMAIN = DomainConfig(
    leaf_type="object",
    node_id_prefix="object:",
    target_noun="object",
    domain_description="concrete objects (sports, animals, fruits, vehicles, etc.)",
)

DISEASES_DOMAIN = DomainConfig(
    leaf_type="disease",
    node_id_prefix="disease:",
    target_noun="disease",
    domain_description="medical conditions (diseases with associated symptoms)",
)
