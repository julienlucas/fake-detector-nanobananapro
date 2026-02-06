chrome.runtime.onInstalled.addListener(() => {
  console.log('FakeFinder extension installed');
});

chrome.action.onClicked.addListener((tab) => {
  chrome.tabs.sendMessage(tab.id, { action: 'activate' });
});
