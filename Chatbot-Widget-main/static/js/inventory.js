document.addEventListener('DOMContentLoaded', () => {
  fetch(`${backend_url}/inventory`)
    .then(response => {
      if (!response.ok) throw new Error('Failed to fetch inventory data');
      return response.json();
    })
    .then(data => populateTable(data, "inventory-table"))
    .catch(error => console.error('Error fetching inventory:', error));
});
