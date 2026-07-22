const express = require('express');
const mongoose = require('mongoose');
const cors = require('cors');
const path = require('path');

// Configuration
const MONGODB_URI = process.env.MONGODB_URI || 'mongodb://localhost:27017';
const MONGODB_DB = process.env.MONGODB_DB || 'IOTA';
const MONGO_URI = `${MONGODB_URI}/${MONGODB_DB}`;
const PORT = process.env.PORT || 3000;

// Connect to MongoDB
mongoose.connect(MONGO_URI)
  .then(() => console.log('Connected to MongoDB'))
  .catch(err => console.error('MongoDB connection error:', err));

// Schemas
const inventorySchema = new mongoose.Schema({
  billing_state: String,
  connectivity_lock: String,
  data_trend: String,
  in_session: String,
  monthly_data: String,
  network_connectivity: String,
  plan_name: String,
  msisdn: { type: Number, unique: true }
}, { collection: 'inventory' });

const customersSchema = new mongoose.Schema({
  customer_type: String,
  name: String,
  agreement_number: Number,
  parent_organization: String
}, { collection: 'customers' });

const subscriptionSchema = new mongoose.Schema({
  imsi: Number,
  Installation_date: Number,
  sim_subscription_state: String,
  msisdn: Number,
  pin1: Number,
  puk1: Number,
  sim_status: String
}, { collection: 'subscription_details' });

// Models
const Inventory = mongoose.model('Inventory', inventorySchema);
const Customers = mongoose.model('Customers', customersSchema);
const Subscription = mongoose.model('Subscription', subscriptionSchema);

// Express app
const app = express();
app.use(cors());
app.use(express.static(path.join(__dirname, '../..')));

// Routes
app.get('/inventory', async (req, res) => {
  try {
    const data = await Inventory.find().lean();
    res.json(data);
  } catch (err) {
    console.error('Error fetching inventory:', err);
    res.status(500).json({ error: 'Internal Server Error' });
  }
});

app.get('/customers', async (req, res) => {
  try {
    const data = await Customers.find().lean();
    res.json(data);
  } catch (err) {
    console.error('Error fetching customers:', err);
    res.status(500).json({ error: 'Internal Server Error' });
  }
});

app.get('/subscriptions', async (req, res) => {
  try {
    const data = await Subscription.find().lean();
    res.json(data);
  } catch (err) {
    console.error('Error fetching subscriptions:', err);
    res.status(500).json({ error: 'Internal Server Error' });
  }
});

app.listen(PORT, () => {
  console.log(`Server running on http://localhost:${PORT}`);
});
