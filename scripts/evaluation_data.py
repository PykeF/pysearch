"""The labelled set used to compare lexical, semantic and hybrid retrieval.

Everything here is synthetic and written for this project, so it carries no
licensing questions — and no claim about retrieval in general. It exists to make
the three modes comparable on the same queries, not to be a benchmark.

Two rules govern it, and they are the whole point:

**The queries were written and fixed before anything was measured.** The corpus
is designed so that different signals should win in different places, not so
that any particular mode comes out ahead.

**The queries are split, and the split is respected.** Parameters — the RRF
constant and the candidate depth — are chosen using the development queries
alone. They are then frozen, and the evaluation queries are used once, with
those frozen values, for the reported numbers. Tuning on the queries you then
report is how a benchmark becomes a advertisement.

Categories
----------

``semantic``  paraphrase, synonym or conceptual: little lexical overlap with
              the answer, which is where exact matching has nothing to work
              with.
``lexical``   an identifier, code, version or rare proper noun decides the
              answer, and the surrounding text is deliberately near-identical
              across several documents, so meaning cannot separate them.
``mixed``     conceptual intent *plus* an exact entity, so both signals hold
              part of the answer.
``distractor`` plausible retrieval ambiguity: documents that are genuinely
              about the same topic as the query but are the wrong answer.
"""

from dataclasses import dataclass

#: Roughly seventy short documents. Groups overlap on vocabulary on purpose:
#: retrieval is only interesting when several documents are plausible.
CORPUS: dict[str, str] = {
    # -- vehicles: described without the words a user would type -------------
    "veh-1": "automobile repair and servicing for engines and gearboxes",
    "veh-2": "keeping a motor vehicle running smoothly through the winter months",
    "veh-3": "brake pads, tyre pressure and other roadworthiness checks",
    "veh-4": "replacing worn windscreen wipers before the rainy season",
    "veh-5": "diagnosing a rattle from the exhaust manifold on older models",
    "veh-6": "an electric drivetrain has far fewer moving parts to service",
    "veh-7": "topping up coolant and checking for leaks around the radiator",
    "veh-8": "gearbox oil should be changed on the interval in the handbook",
    # -- information retrieval ----------------------------------------------
    "ir-1": "BM25 ranks documents using term frequency and inverse document frequency",
    "ir-2": "an inverted index maps each term to the documents that contain it",
    "ir-3": "vector similarity retrieves passages that mean the same thing",
    "ir-4": "tokenization splits text into the units an index actually stores",
    "ir-5": "stop words carry little signal but occupy most of a posting list",
    "ir-6": "stemming collapses inflected forms so that they share one index entry",
    "ir-7": "precision and recall pull against each other as the cut-off moves",
    "ir-8": "reciprocal rank fusion combines two rankings without comparing their scores",
    # -- distributed systems -------------------------------------------------
    "sys-1": "sharding splits a dataset so that each node stores only part of it",
    "sys-2": "replication keeps a second durable copy so a lost node loses nothing",
    "sys-3": "a coordinator fans a query out to every shard and merges the answers",
    "sys-4": "a write is acknowledged only once every copy has committed it durably",
    "sys-5": "leader election lets a cluster agree on which node may accept writes",
    "sys-6": "a partition forces a choice between answering and staying consistent",
    "sys-7": "back pressure protects a service from accepting more than it can finish",
    "sys-8": "idempotent operations make retries safe when a response is lost",
    # -- cooking, as an unrelated topic --------------------------------------
    "cook-1": "simmer the sauce gently while the pasta finishes cooking",
    "cook-2": "a sharp knife and a hot pan matter more than an expensive recipe",
    "cook-3": "let the dough rest before shaping it into loaves",
    "cook-4": "season in layers rather than all at the end",
    "cook-5": "a heavy base spreads heat evenly and stops things catching",
    "cook-6": "cold butter is what makes pastry flake rather than toughen",
    # -- error codes: near-identical apart from the code ----------------------
    "err-1": "error ERR_CONN_RESET_1042 means the peer closed the connection mid-request",
    "err-2": "error ERR_CONN_RESET_1043 means the peer closed the connection mid-request",
    "err-3": "error ERR_CONN_RESET_1044 means the peer closed the connection mid-request",
    "err-4": "error ERR_DISK_FULL_2210 means the write path ran out of space",
    "err-5": "error ERR_DISK_FULL_2211 means the write path ran out of space",
    "err-6": "error ERR_CHECKSUM_3301 means a page failed verification on read",
    # -- part numbers: near-identical apart from the part ---------------------
    "part-1": "PX-9174-Q battery replacement procedure for field engineers",
    "part-2": "PX-9175-Q battery replacement procedure for field engineers",
    "part-3": "PX-9174-Q display calibration procedure for field engineers",
    "part-4": "PX-9176-Q battery replacement procedure for field engineers",
    "part-5": "QT-4420-B fan assembly removal and refitting instructions",
    "part-6": "QT-4421-B fan assembly removal and refitting instructions",
    # -- standards references -------------------------------------------------
    "rfc-1": "RFC 9110 defines the semantics of HTTP messages and methods",
    "rfc-2": "RFC 9111 covers HTTP caching and how responses may be reused",
    "rfc-3": "RFC 8446 specifies version 1.3 of the transport layer security protocol",
    "rfc-4": "RFC 7519 describes JSON web tokens and how they are validated",
    "rfc-5": "RFC 6265 explains how cookies are set and returned by user agents",
    # -- versions: near-identical apart from the version ----------------------
    "ver-1": "release 4.2.0 removes the deprecated batch import endpoint",
    "ver-2": "release 4.3.0 removes the deprecated batch import endpoint",
    "ver-3": "release 4.2.1 fixes a regression in the batch import endpoint",
    "ver-4": "release 5.0.0 rewrites the batch import endpoint entirely",
    # -- product and project names --------------------------------------------
    "prod-1": "Halberd is the internal scheduler that drains nodes before maintenance",
    "prod-2": "Marlinspike is the deployment tool that promotes builds between stages",
    "prod-3": "Tanglefoot is the rate limiter that sits in front of the public API",
    "prod-4": "Blackthorn is the archival service that moves cold data off primaries",
    "prod-5": "Wintergreen is the dashboard that operators watch during an incident",
    # -- operations, sharing vocabulary with several groups -------------------
    "ops-1": "restarting a node drains its traffic before the process exits",
    "ops-2": "a rolling upgrade replaces one instance at a time to avoid downtime",
    "ops-3": "disk pressure is the most common cause of a failed write on this fleet",
    "ops-4": "connection resets during deploys are usually the load balancer, not the app",
    "ops-5": "check the battery health report before scheduling a field visit",
    "ops-6": "calibration drifts slowly, so displays are checked at every service",
    # -- general distractors ---------------------------------------------------
    "gen-1": "the office moves to the new building at the end of the quarter",
    "gen-2": "expenses must be submitted within thirty days of the purchase",
    "gen-3": "the fire drill happens twice a year and takes about ten minutes",
    "gen-4": "parking permits are issued per person rather than per vehicle",
    "gen-5": "the canteen stops serving hot food at two in the afternoon",
}


@dataclass(frozen=True, slots=True)
class LabelledQuery:
    """A query, the documents a human would accept, and why it is here."""

    query: str
    relevant: frozenset[str]
    category: str
    note: str


#: Used only to choose the RRF constant and the candidate depth. Never reported
#: as an evaluation result.
DEVELOPMENT_QUERIES: tuple[LabelledQuery, ...] = (
    LabelledQuery(
        "car maintenance",
        frozenset({"veh-1", "veh-2", "veh-3", "veh-4", "veh-7", "veh-8"}),
        "semantic",
        "'car' appears in no document",
    ),
    LabelledQuery(
        "keeping food from sticking to the pan",
        frozenset({"cook-5", "cook-2"}),
        "semantic",
        "conceptual, no shared content words",
    ),
    LabelledQuery(
        "what happens when the network splits",
        frozenset({"sys-6"}),
        "semantic",
        "paraphrase of a partition",
    ),
    LabelledQuery(
        "ERR_DISK_FULL_2210",
        frozenset({"err-4"}),
        "lexical",
        "code among a near-identical pair",
    ),
    LabelledQuery(
        "RFC 8446",
        frozenset({"rfc-3"}),
        "lexical",
        "standard number among other RFCs",
    ),
    LabelledQuery(
        "Tanglefoot",
        frozenset({"prod-3"}),
        "lexical",
        "rare proper noun",
    ),
    LabelledQuery(
        "release 4.2.1",
        frozenset({"ver-3"}),
        "lexical",
        "version among near-identical releases",
    ),
    LabelledQuery(
        "PX-9175-Q battery swap",
        frozenset({"part-2"}),
        "mixed",
        "exact part plus a paraphrased action",
    ),
    LabelledQuery(
        "ERR_CONN_RESET_1042 during a deploy",
        frozenset({"err-1"}),
        "mixed",
        "exact code plus operational context",
    ),
    LabelledQuery(
        "batch import endpoint changes",
        frozenset({"ver-1", "ver-2", "ver-3", "ver-4"}),
        "distractor",
        "four near-identical release notes are all relevant",
    ),
    LabelledQuery(
        "fan assembly instructions",
        frozenset({"part-5", "part-6"}),
        "distractor",
        "two parts share the whole description",
    ),
    LabelledQuery(
        "why did the connection drop",
        frozenset({"err-1", "err-2", "err-3", "ops-4"}),
        "distractor",
        "several documents describe the same symptom",
    ),
)

#: Held out. Measured once, with the parameters already frozen.
EVALUATION_QUERIES: tuple[LabelledQuery, ...] = (
    # -- semantic ------------------------------------------------------------
    LabelledQuery(
        "fixing a broken engine",
        frozenset({"veh-1", "veh-5", "veh-8"}),
        "semantic",
        "paraphrase with partial overlap",
    ),
    LabelledQuery(
        "searching by meaning rather than keywords",
        frozenset({"ir-3"}),
        "semantic",
        "conceptual, little lexical overlap",
    ),
    LabelledQuery(
        "surviving the loss of a machine",
        frozenset({"sys-2"}),
        "semantic",
        "paraphrase of replication",
    ),
    LabelledQuery(
        "splitting data across machines",
        frozenset({"sys-1"}),
        "semantic",
        "paraphrase of sharding",
    ),
    LabelledQuery(
        "making dinner",
        frozenset({"cook-1", "cook-2", "cook-3", "cook-4", "cook-5", "cook-6"}),
        "semantic",
        "topic paraphrase",
    ),
    LabelledQuery(
        "making retries safe",
        frozenset({"sys-8"}),
        "semantic",
        "paraphrase of idempotence",
    ),
    # -- lexical -------------------------------------------------------------
    LabelledQuery(
        "ERR_CHECKSUM_3301",
        frozenset({"err-6"}),
        "lexical",
        "unique code",
    ),
    LabelledQuery(
        "ERR_CONN_RESET_1044",
        frozenset({"err-3"}),
        "lexical",
        "code among three near-identical documents",
    ),
    LabelledQuery(
        "PX-9176-Q",
        frozenset({"part-4"}),
        "lexical",
        "part among three near-identical documents",
    ),
    LabelledQuery(
        "RFC 7519",
        frozenset({"rfc-4"}),
        "lexical",
        "standard number",
    ),
    LabelledQuery(
        "Blackthorn",
        frozenset({"prod-4"}),
        "lexical",
        "rare proper noun",
    ),
    LabelledQuery(
        "release 5.0.0",
        frozenset({"ver-4"}),
        "lexical",
        "version among near-identical releases",
    ),
    # -- mixed ---------------------------------------------------------------
    LabelledQuery(
        "PX-9174-Q battery failure",
        frozenset({"part-1"}),
        "mixed",
        "exact part; a sibling part and a sibling procedure both distract",
    ),
    LabelledQuery(
        "Halberd node draining",
        frozenset({"prod-1"}),
        "mixed",
        "proper noun plus a concept another document also describes",
    ),
    LabelledQuery(
        "RFC 9111 response reuse",
        frozenset({"rfc-2"}),
        "mixed",
        "standard number plus a paraphrase of its subject",
    ),
    LabelledQuery(
        "combining two rankings",
        frozenset({"ir-8"}),
        "mixed",
        "concept stated in the document's own words",
    ),
    # -- distractor ----------------------------------------------------------
    LabelledQuery(
        "battery replacement procedure",
        frozenset({"part-1", "part-2", "part-4"}),
        "distractor",
        "three parts share the procedure; a fourth document is calibration",
    ),
    LabelledQuery(
        "running out of disk space",
        frozenset({"err-4", "err-5", "ops-3"}),
        "distractor",
        "codes and an operational note describe the same failure",
    ),
    LabelledQuery(
        "display calibration",
        frozenset({"part-3", "ops-6"}),
        "distractor",
        "one part document and one operational note",
    ),
    LabelledQuery(
        "upgrading without downtime",
        frozenset({"ops-2", "ops-1"}),
        "distractor",
        "two operational notes, one only partly on topic",
    ),
)

CATEGORIES: tuple[str, ...] = ("semantic", "lexical", "mixed", "distractor")
