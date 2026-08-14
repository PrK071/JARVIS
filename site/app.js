const form = document.querySelector("#signup-form");
const statusMessage = document.querySelector("#form-status");

const fields = {
  name: document.querySelector("#name"),
  email: document.querySelector("#email"),
  password: document.querySelector("#password"),
  confirmation: document.querySelector("#password-confirmation"),
  terms: document.querySelector("#terms"),
};

const errors = {
  name: document.querySelector("#name-error"),
  email: document.querySelector("#email-error"),
  password: document.querySelector("#password-error"),
  confirmation: document.querySelector("#password-confirmation-error"),
  terms: document.querySelector("#terms-error"),
};

const emailPattern = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
const passwordPattern = /^(?=.*[a-z])(?=.*[A-Z])(?=.*\d).{8,}$/;

function setError(fieldName, message) {
  const field = fields[fieldName];
  field.setAttribute("aria-invalid", message ? "true" : "false");
  errors[fieldName].textContent = message;
  return !message;
}

function validateName() {
  const value = fields.name.value.trim();
  if (!value) return setError("name", "Informe seu nome completo.");
  if (value.length < 3) return setError("name", "Use pelo menos 3 caracteres.");
  return setError("name", "");
}

function validateEmail() {
  const value = fields.email.value.trim();
  if (!value) return setError("email", "Informe seu e-mail.");
  if (!emailPattern.test(value)) return setError("email", "Digite um e-mail válido.");
  return setError("email", "");
}

function validatePassword() {
  const value = fields.password.value;
  if (!value) return setError("password", "Crie uma senha.");
  if (!passwordPattern.test(value)) {
    return setError("password", "A senha deve ter 8 caracteres, maiúscula, minúscula e número.");
  }
  return setError("password", "");
}

function validateConfirmation() {
  if (!fields.confirmation.value) return setError("confirmation", "Confirme sua senha.");
  if (fields.confirmation.value !== fields.password.value) {
    return setError("confirmation", "As senhas não coincidem.");
  }
  return setError("confirmation", "");
}

function validateTerms() {
  return fields.terms.checked
    ? setError("terms", "")
    : setError("terms", "Você precisa aceitar os termos para continuar.");
}

const validators = {
  name: validateName,
  email: validateEmail,
  password: validatePassword,
  confirmation: validateConfirmation,
  terms: validateTerms,
};

Object.entries(fields).forEach(([fieldName, field]) => {
  const eventName = field.type === "checkbox" ? "change" : "blur";
  field.addEventListener(eventName, validators[fieldName]);

  if (field.type !== "checkbox") {
    field.addEventListener("input", () => {
      if (field.getAttribute("aria-invalid") === "true") validators[fieldName]();
      if (fieldName === "password" && fields.confirmation.value) validateConfirmation();
      statusMessage.textContent = "";
    });
  }
});

document.querySelectorAll("[data-toggle-password]").forEach((button) => {
  button.addEventListener("click", () => {
    const target = document.querySelector(`#${button.dataset.togglePassword}`);
    const shouldShow = target.type === "password";
    target.type = shouldShow ? "text" : "password";
    button.textContent = shouldShow ? "Ocultar" : "Mostrar";
    button.setAttribute("aria-pressed", String(shouldShow));
    button.setAttribute("aria-label", `${shouldShow ? "Ocultar" : "Mostrar"} ${target.id === "password" ? "senha" : "confirmação de senha"}`);
  });
});

form.addEventListener("submit", (event) => {
  event.preventDefault();
  statusMessage.textContent = "";

  const results = Object.values(validators).map((validate) => validate());
  if (results.includes(false)) {
    const firstInvalidField = Object.values(fields).find(
      (field) => field.getAttribute("aria-invalid") === "true",
    );
    firstInvalidField?.focus();
    return;
  }

  statusMessage.textContent = "Dados validados com sucesso. Seu acesso está pronto para ser conectado ao serviço de identidade.";
});
