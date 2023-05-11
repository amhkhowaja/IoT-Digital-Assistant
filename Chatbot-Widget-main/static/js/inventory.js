
document.addEventListener('DOMContentLoaded', () => {
  fetch('http://localhost:3000/inventory')
    .then(response => {
      if (response.status !== 200) {
        throw new Error('Failed to fetch inventory data');
      }
      return response.json();
    })
    .then(inventoryData => {
      console.log(inventoryData);
      populateTable(inventoryData, "inventory-table");
    })
    .catch(error => {
      console.error('Error fetching inventory data:', error);
    });
});