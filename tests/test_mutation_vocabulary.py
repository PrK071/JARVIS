from __future__ import annotations

import pytest

from tern.orchestrator.decision_policy import AgentDecisionPolicy, Intent
from tern.orchestrator.explicit_agent_binding import detect_explicit_agent_binding
from tern.orchestrator.intent_semantics import MutationAction, classify_mutation_action


DELETE_OBJECT_PHRASES = (
    "deletar", "delete", "deleta", "delete isso", "deletar isso",
    "deletar esse", "deletar essa", "apagar", "apaga", "apague",
    "apagar isso", "apaga isso", "apague isso", "excluir", "exclui",
    "exclua", "excluir isso", "exclui isso", "exclua isso", "remover",
    "remove", "remova", "retirar", "retira", "retire", "eliminar",
    "elimina", "elimine", "suprimir", "suprime", "suprima", "descartar",
    "descarta", "descarte", "extinguir", "extingue", "extinga",
    "erradicar", "erradica", "dar baixa", "baixar", "dar baixa nisso",
    "delete o arquivo", "apague o arquivo", "exclua o arquivo",
    "remova o arquivo", "elimine o arquivo", "delete a pasta",
    "apague a pasta", "exclua a pasta", "remova a pasta",
    "remova o diretório", "delete o diretório", "apague o diretório",
    "exclua o registro", "remova o registro", "delete o registro",
    "apague a entrada", "remova a entrada", "exclua a entrada",
    "remova o recurso", "delete o recurso", "exclua o recurso",
    "desinstale", "desinstala", "remova a instalação", "tira isso",
    "tira daí", "tira fora", "remove isso daí", "joga fora",
    "manda embora", "some com isso", "faz isso sumir", "faz desaparecer",
    "faz isso desaparecer", "se livra disso", "livra disso",
    "não quero mais isso", "não precisa mais disso", "pode tirar",
    "pode apagar", "pode excluir", "pode deletar", "pode remover",
    "mete o delete", "dá delete", "dá um delete nisso", "apaga daí",
    "apaga tudo", "arranca isso", "corta isso", "corta fora",
    "remove de vez", "elimina de vez", "apaga de vez", "exclui de vez",
    "deleta de vez", "delete o registro", "apague o registro",
    "exclua o registro", "remova o registro", "delete os dados",
    "apague os dados", "remova os dados", "exclua da base",
    "remova do banco", "delete do banco", "apague do banco",
    "remova da tabela", "delete da tabela", "exclua da tabela",
    "apague do histórico", "remova do histórico", "delete o histórico",
    "desative e remova", "isso não deve mais existir",
    "isso não deveria existir", "isso pode deixar de existir",
    "quero isso fora", "quero isso removido", "quero isso apagado",
    "quero isso excluído", "quero isso deletado", "não mantenha isso",
    "não preserve isso", "não deixe isso aí", "não deixe isso no código",
    "não deixe isso no projeto", "tire isso do projeto",
    "quero isso fora do projeto", "isso tem que sair", "isso precisa sair",
    "isso deve sair",
)

REMOVE_COMPONENT_PHRASES = (
    "remova esse trecho", "apague esse trecho", "delete essa linha",
    "exclua essa parte", "tire essa parte", "corte esse trecho",
    "elimine essa seção", "retire esse campo", "remova essa propriedade",
    "tire esse parâmetro", "apague esse valor", "apaga o campo description",
    "remove esse código", "apaga esse código", "exclui esse código",
    "delete esse código", "tira esse código", "corta esse código",
    "remove essa função", "apaga essa função", "delete essa função",
    "exclua esse método", "remova essa classe", "tire esse import",
    "remova essa dependência", "apague essa variável", "remova esse bloco",
    "corte esse trecho", "elimine essa lógica", "retire essa implementação",
    "remova essa configuração", "apague essa configuração",
    "exclua essa configuração", "delete essa configuração",
    "retire essa opção", "tire essa configuração", "elimine essa regra",
    "remova essa regra", "apague essa regra", "remova do config",
    "delete do config", "tire do config",
)

CLEAR_CONTENT_PHRASES = (
    "limpar", "limpa", "limpe", "esvaziar", "esvazia", "esvazie",
    "clear", "dar clear", "limpar tudo", "limpar conteúdo",
    "apagar conteúdo", "esvaziar conteúdo", "remover conteúdo",
    "limpa o arquivo", "limpe esse campo", "esvazie esse campo",
    "limpe os dados", "limpe o histórico", "limpa isso", "limpa daí",
)

RESET_STATE_PHRASES = (
    "zerar", "zera", "zere", "resetar", "reseta", "resete", "reset",
    "zera isso", "zera essa parte", "zere esse valor",
)


@pytest.mark.parametrize(
    ("action", "phrases"),
    [
        (MutationAction.DELETE_OBJECT, DELETE_OBJECT_PHRASES),
        (MutationAction.REMOVE_COMPONENT, REMOVE_COMPONENT_PHRASES),
        (MutationAction.CLEAR_CONTENT, CLEAR_CONTENT_PHRASES),
        (MutationAction.RESET_STATE, RESET_STATE_PHRASES),
    ],
)
def test_mutation_dataset_converges_to_distinct_actions(action, phrases):
    failures = {
        phrase: classify_mutation_action(phrase)
        for phrase in phrases
        if classify_mutation_action(phrase) is not action
    }
    assert failures == {}


@pytest.mark.parametrize(
    "text",
    [
        "não delete o arquivo",
        "não deleta o arquivo",
        "não apague essa função",
        "não apaga essa função",
        "não remova o campo description",
        "não remove o campo description",
        "sem limpar o arquivo",
        "não limpa o arquivo",
        "não resete essa configuração",
        "não reseta essa configuração",
    ],
)
def test_negated_mutation_commands_never_execute(text):
    assert classify_mutation_action(text) is None
    decision = AgentDecisionPolicy().decide(
        text,
        fixture_context={"active_project": "tern"},
    )
    assert decision.intent not in {Intent.CODEX_DELEGATE, Intent.CODEX_STEER}


@pytest.mark.parametrize(
    ("text", "action"),
    [
        ("delete o arquivo", MutationAction.DELETE_OBJECT),
        ("remove essa função", MutationAction.REMOVE_COMPONENT),
        ("limpa o arquivo", MutationAction.CLEAR_CONTENT),
        ("zera essa configuração", MutationAction.RESET_STATE),
    ],
)
def test_mutation_action_is_exposed_on_the_delegation_decision(text, action):
    decision = AgentDecisionPolicy().decide(
        text,
        fixture_context={"active_project": "tern"},
    )
    assert decision.intent is Intent.CODEX_DELEGATE
    assert decision.tools == ("delegate_to_codex",)
    assert decision.intent_frame is not None
    assert decision.intent_frame.action is action
    assert decision.as_dict()["action"] == action.value


@pytest.mark.parametrize(
    ("text", "focused_agent", "action"),
    [
        ("manda o Codex apagar isso", None, MutationAction.DELETE_OBJECT),
        ("pede pro Codex excluir isso", None, MutationAction.DELETE_OBJECT),
        ("faz o Codex deletar isso", None, MutationAction.DELETE_OBJECT),
        ("passa pro Codex remover essa função", None, MutationAction.REMOVE_COMPONENT),
        ("bota o Codex pra tirar isso", None, MutationAction.DELETE_OBJECT),
        ("deixa o Codex limpar isso", None, MutationAction.CLEAR_CONTENT),
        ("manda ele excluir essa parte", "codex", MutationAction.REMOVE_COMPONENT),
        ("pede pra ele remover esse arquivo", "codex", MutationAction.DELETE_OBJECT),
        ("faz ele apagar esses registros", "codex", MutationAction.DELETE_OBJECT),
        ("manda ele se livrar disso", "codex", MutationAction.DELETE_OBJECT),
    ],
)
def test_delegated_mutations_keep_agent_and_action(text, focused_agent, action):
    binding = detect_explicit_agent_binding(text, focused_agent=focused_agent)
    assert binding is not None
    assert binding.requested_agent == "codex"
    decision = AgentDecisionPolicy().decide(
        text,
        fixture_context={"active_project": "tern", "focused_agent": focused_agent},
        explicit_agent_binding=binding,
    )
    assert decision.intent is Intent.CODEX_DELEGATE
    assert decision.intent_frame is not None
    assert decision.intent_frame.action is action
