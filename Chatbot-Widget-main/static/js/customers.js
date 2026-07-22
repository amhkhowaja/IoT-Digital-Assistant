document.addEventListener('DOMContentLoaded', () => {
  fetch('http://localhost:3000/customers')
    .then(response => {
      if (response.status !== 200) {
        throw new Error('Failed to fetch customer data');
      }
      return response.json();
    })
    .then(customersData => {
      populateTable(customersData, "customers-table");
    })
    .catch(error => {
      console.error('Error fetching customers data:', error);
    });
});