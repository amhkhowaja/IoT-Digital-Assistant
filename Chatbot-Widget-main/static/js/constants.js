// Configuration
const rasa_server_url = "http://localhost:5005/webhooks/rest/webhook";
const backend_url = "http://localhost:3000";
const sender_id = (typeof uuidv4 === 'function') ? uuidv4() : Math.random().toString(36).slice(2);
