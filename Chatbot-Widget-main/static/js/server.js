// Required dependencies
const express = require('express');
const { Decimal128 } = require('mongodb');
const mongoose = require('mongoose');
const cors = require('cors');
// const mongooseLong = require('mongoose-long');


// Connect to MongoDB
mongoose.connect('mongodb://localhost/IOTA', { useNewUrlParser: true, useUnifiedTopology: true })
  .then(() => console.log('Connected to MongoDB'))
  .catch(err => console.error('Error connecting to MongoDB:', err));

// Create a Mongoose schema for the inventory collection
const inventorySchema = new mongoose.Schema({
  _id: mongoose.Schema.Types.ObjectId,
  billing_state: String,
  connectivity_lock: String,
  data_trend: String,
  in_session: String,
  monthly_data: String,
  network_connectivity: String,
  plan_name: String,
  msisdn: { type: Number, unique: true }
});
const customersSchema = new mongoose.Schema({
  _id: mongoose.Schema.Types.ObjectId,
  customer_type: String,
  name: String,
  agreement_number: Number,
  parent_organization: String
});
const subscriptionSchema = new mongoose.Schema({
  _id: mongoose.Schema.Types.ObjectId,
  imsi: Number,
  Installation_date: Number,
  sim_subscription_state: String,
  msisdn: Number,
  pin1: Number,
  puk1: Number,
  sim_status: String
  
});

// Create a Mongoose model based on the schema
const Inventory = mongoose.model('Inventory', inventorySchema, 'inventory');
const Customers = mongoose.model('Customers', customersSchema, 'customers');
const Subscription = mongoose.model('Subscription', subscriptionSchema, 'subscription_details');

// Create an Express application
const app = express();
app.use(cors());
// Define a route to retrieve inventory data and populate the HTML table
app.get('/inventory/', async (req, res) => {
  try {
    // Retrieve inventory data from the database
    const inventoryData = await Inventory.find();
    res.send(inventoryData);
  } catch (err) {
    console.error('Error retrieving inventory data:', err);
    res.status(500).send('Internal Server Error');
  }
  console.log("/inventory/ Endpoint is running");

});
app.get('/inventory/', async (req, res) => {
  try {
    filter=req.query.filter;
    // Retrieve inventory data from the database
    console.log(filter);
    const inventoryData = await Inventory.find(filter);
    res.send(inventoryData);
  } catch (err) {
    console.error('Error retrieving inventory data:', err);
    res.status(500).send('Internal Server Error');
  }
  console.log("/inventory/ Endpoint is running");

});


app.get('/customers/', async (req, res) => {
  try {
    // Retrieve inventory data from the database
    const customersData = await Customers.find();
    res.send(customersData);
  } catch (err) {
    console.error('Error retrieving customers data:', err);
    res.status(500).send('Internal Server Error');
  }
  console.log("/customers/ Endpoint is running");
});
app.get('/subscriptions/', async (req, res) => {
  try {
    // Retrieve inventory data from the database
    const subscriptionData = await Subscription.find();
    res.send(subscriptionData);
  } catch (err) {
    console.error('Error retrieving customers data:', err);
    res.status(500).send('Internal Server Error');
    
  }
  console.log("/subscriptions/ Endpoint is running");
});
// Start the server
app.listen(3000, () => {
  console.log('Server listening on port 3000');
});
