# Security

PySearch is an **educational portfolio project**, not production software. It
has not undergone a security audit, and no vulnerability-response SLA is
offered. Please treat it accordingly.

## Trusted-network assumption

The `/internal/*` endpoints — node status, statistics, scoring, replication and
corpus export — **have no authentication of any kind**. Any client that can
reach a node can read its entire corpus through `/internal/export`, or write to
it through the replication endpoints.

This is a deliberate scope boundary, not an oversight: the project explores
retrieval and distributed-systems problems, and authentication would add
surface without adding insight into those. It is stated plainly rather than
implied.

**Do not expose PySearch nodes directly to the public internet.** A deployment
would need, at minimum:

- internal endpoints reachable only on a private network
- authentication and TLS terminated in front of every node
- the coordinator as the only externally reachable process

None of that is implemented here.

## Also absent

- No authentication or authorization on the public API either
- No TLS anywhere; all inter-node traffic is plain HTTP
- No rate limiting, quotas, or request-size limits beyond FastAPI's defaults
- No audit logging of who changed what
- No secret management (there are no secrets: `.env.example` documents every
  setting, and none of them is a credential)

## Reporting

If you find a security problem, please open a GitHub issue. Since this project
is not deployed anywhere and holds no user data, ordinary public disclosure is
appropriate — there is no production system at risk. If you would rather not
post publicly, say so in an issue without details and a private channel can be
arranged.
