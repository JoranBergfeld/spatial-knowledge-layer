// The layer's prod Foundry: one AIServices account and its two deployments
// (personal-website #90). Visitor traffic only. KIT builds call the bench's
// spatial-foundry under the same deployment names (#91), so build spend never
// lands here without a ledger row.
//
// Deploy, from the repo root, into the Visual Studio Enterprise subscription:
//
//   az deployment group what-if -g rg-spatial-knowledge-layer -f infra/main.bicep
//   az deployment group create  -g rg-spatial-knowledge-layer -f infra/main.bicep
//
// No Search here. The layer's search is pgvector on homelab's Postgres (#96).

@description('The account name, also its subdomain: https://<name>.cognitiveservices.azure.com/')
param name string = 'spatial-prod-foundry'

param location string = 'swedencentral'

// Global, because that's what the site's app.budget.rates prices. Capacity is
// in thousands of tokens per minute. Nothing here embeds: the layer embeds
// with google/embeddinggemma-2 on the homelab's Spark since snapshot 2026-11-21.
// A snapshot embedded with text-embedding-3-small can no longer be served.
// Versions upgrade on their own, same as the bench: a retired pin takes the
// layer down, a newer default doesn't. what-if will show the version moving.
var deployments = [
  {
    name: 'gpt-5.4' // plans
    model: 'gpt-5.4'
    version: '2026-03-05'
    capacity: 100
  }
  {
    name: 'gpt-5.4-mini' // composes
    model: 'gpt-5.4-mini'
    version: '2026-03-17'
    capacity: 200
  }
]

// Public endpoint, key auth on, no IP rules. A home IP changes, and a firewall
// would take the layer offline with nothing saying why. The key is the guard.
resource account 'Microsoft.CognitiveServices/accounts@2024-10-01' = {
  name: name
  location: location
  kind: 'AIServices'
  sku: {
    name: 'S0'
  }
  properties: {
    customSubDomainName: name
    publicNetworkAccess: 'Enabled'
    disableLocalAuth: false
  }
}

// One at a time. Azure rejects concurrent deployment writes on one account.
@batchSize(1)
resource deployment 'Microsoft.CognitiveServices/accounts/deployments@2024-10-01' = [
  for d in deployments: {
    parent: account
    name: d.name
    sku: {
      name: 'GlobalStandard'
      capacity: d.capacity
    }
    properties: {
      model: {
        format: 'OpenAI'
        name: d.model
        version: d.version
      }
      versionUpgradeOption: 'OnceNewDefaultVersionAvailable'
    }
  }
]

output endpoint string = account.properties.endpoint
