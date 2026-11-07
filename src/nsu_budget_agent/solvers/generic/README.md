# Общий маршрут — B

Stateless `assess/step` по Contracts v1, а не самостоятельный внешний цикл.
Один step возвращает ToolCall, CandidateAnswer или StopAttempt и делает не более
одного complete через LLMSession A. История передаётся как tuple StepRecord.
Инструменты, переходы, лимиты и AttemptResult исполняет/создаёт JiuwenAdapter A.
Неизвестные типы допустимы; собственных HTTP-клиента, публикации или SDK-импортов нет.
