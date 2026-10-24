-- Social Follow Graph Analytics (Apache Spark SQL)
-- Source: analytics.follows(user_id, broadcaster_id)
-- Assumption: source contains CURRENT ACTIVE follow relationships.
-- If source contains follow/unfollow EVENTS, reconstruct the latest state per
-- (user_id, broadcaster_id) using event timestamp and deterministic tie-breaker,
-- retaining only pairs whose latest action is 'follow', BEFORE running this file.
-- This script uses Spark SQL temporary views (session-scoped). Views reused by
-- several downstream steps are cached (MEMORY_AND_DISK) so their source scans
-- and shuffles run once; UNCACHE them at the end of the session if needed.
-- Large datasets: broadcaster pair generation is O(sum_user following_count^2).
-- Filter high-degree users or precompute top-K pairs for production use.

-- 1. Deduplicate edges, namespace node identifiers if exporting to a graph engine.
CREATE OR REPLACE TEMP VIEW follow_edges AS
SELECT DISTINCT
  CAST(user_id AS STRING) AS user_id,
  CAST(broadcaster_id AS STRING) AS broadcaster_id
FROM analytics.follows
WHERE user_id IS NOT NULL AND broadcaster_id IS NOT NULL;
CACHE TABLE follow_edges;

-- 2. Overall graph statistics (counts exclude isolated nodes absent from edges).
CREATE OR REPLACE TEMP VIEW graph_summary AS
SELECT
  COUNT(*) AS total_edges,
  COUNT(DISTINCT user_id) AS total_users,
  COUNT(DISTINCT broadcaster_id) AS total_broadcasters,
  CAST(COUNT(*) AS DOUBLE) / NULLIF(COUNT(DISTINCT user_id), 0) AS avg_follows_per_user,
  CAST(COUNT(*) AS DOUBLE) / NULLIF(COUNT(DISTINCT broadcaster_id), 0) AS avg_followers_per_broadcaster
FROM follow_edges;

-- 3. Node degrees.
CREATE OR REPLACE TEMP VIEW broadcaster_degrees AS
SELECT broadcaster_id, COUNT(*) AS follower_count
FROM follow_edges GROUP BY broadcaster_id;

CREATE OR REPLACE TEMP VIEW user_degrees AS
SELECT user_id, COUNT(*) AS following_count
FROM follow_edges GROUP BY user_id;

-- 4. Degree distributions, useful for skew and long-tail profiling.
CREATE OR REPLACE TEMP VIEW broadcaster_degree_distribution AS
SELECT follower_count, COUNT(*) AS broadcaster_count
FROM broadcaster_degrees GROUP BY follower_count;

CREATE OR REPLACE TEMP VIEW user_degree_distribution AS
SELECT following_count, COUNT(*) AS user_count
FROM user_degrees GROUP BY following_count;

-- 5. Undirected broadcaster similarity pairs based on shared followers.
-- IMPORTANT: expensive for users following many broadcasters.
CREATE OR REPLACE TEMP VIEW broadcaster_pairs AS
SELECT
  a.broadcaster_id AS broadcaster_a,
  b.broadcaster_id AS broadcaster_b,
  COUNT(*) AS shared_followers
FROM follow_edges a
JOIN follow_edges b
  ON a.user_id = b.user_id
 AND a.broadcaster_id < b.broadcaster_id
GROUP BY a.broadcaster_id, b.broadcaster_id;

-- 6. Jaccard similarity between broadcaster follower sets.
CREATE OR REPLACE TEMP VIEW broadcaster_similarity AS
SELECT
  p.broadcaster_a,
  p.broadcaster_b,
  p.shared_followers,
  da.follower_count AS followers_a,
  db.follower_count AS followers_b,
  CAST(p.shared_followers AS DOUBLE) /
    NULLIF(da.follower_count + db.follower_count - p.shared_followers, 0) AS jaccard_similarity
FROM broadcaster_pairs p
JOIN broadcaster_degrees da ON p.broadcaster_a = da.broadcaster_id
JOIN broadcaster_degrees db ON p.broadcaster_b = db.broadcaster_id
WHERE p.shared_followers >= 2;
-- Read by sections 7-11; caching avoids recomputing the quadratic pair join.
CACHE TABLE broadcaster_similarity;

-- 7. Make similarity directional for item-based collaborative filtering.
CREATE OR REPLACE TEMP VIEW broadcaster_neighbors AS
SELECT broadcaster_a AS source_broadcaster,
       broadcaster_b AS candidate_broadcaster,
       shared_followers, jaccard_similarity
FROM broadcaster_similarity
UNION ALL
SELECT broadcaster_b AS source_broadcaster,
       broadcaster_a AS candidate_broadcaster,
       shared_followers, jaccard_similarity
FROM broadcaster_similarity;

-- 8. Keep top 20 neighbors per broadcaster to reduce recommendation joins.
CREATE OR REPLACE TEMP VIEW broadcaster_top_neighbors AS
SELECT source_broadcaster, candidate_broadcaster,
       shared_followers, jaccard_similarity
FROM (
  SELECT n.*,
         ROW_NUMBER() OVER (
           PARTITION BY source_broadcaster
           ORDER BY jaccard_similarity DESC, shared_followers DESC,
                    candidate_broadcaster ASC
         ) AS rn
  FROM broadcaster_neighbors n
) ranked
WHERE rn <= 20;

-- 9. Candidate scoring, excluding already-followed broadcasters.
CREATE OR REPLACE TEMP VIEW recommendation_candidates AS
SELECT
  f.user_id,
  n.candidate_broadcaster AS broadcaster_id,
  SUM(n.jaccard_similarity) AS recommendation_score,
  COUNT(*) AS supporting_follows
FROM follow_edges f
JOIN broadcaster_top_neighbors n
  ON f.broadcaster_id = n.source_broadcaster
LEFT ANTI JOIN follow_edges existing
  ON f.user_id = existing.user_id
 AND n.candidate_broadcaster = existing.broadcaster_id
GROUP BY f.user_id, n.candidate_broadcaster;

-- 10. Top 10 recommendations per user.
CREATE OR REPLACE TEMP VIEW top10_broadcaster_recommendations AS
SELECT user_id, broadcaster_id, recommendation_score, supporting_follows
FROM (
  SELECT c.*,
         ROW_NUMBER() OVER (
           PARTITION BY user_id
           ORDER BY recommendation_score DESC,
                    supporting_follows DESC, broadcaster_id ASC
         ) AS rn
  FROM recommendation_candidates c
) ranked
WHERE rn <= 10;

-- 11. Inspect results. Replace LIMITs or persist views as tables as needed.
SELECT * FROM graph_summary;
SELECT * FROM broadcaster_degrees ORDER BY follower_count DESC LIMIT 20;
SELECT * FROM user_degrees ORDER BY following_count DESC LIMIT 20;
SELECT * FROM broadcaster_degree_distribution ORDER BY follower_count DESC LIMIT 20;
SELECT * FROM user_degree_distribution ORDER BY following_count DESC LIMIT 20;
SELECT * FROM broadcaster_similarity ORDER BY jaccard_similarity DESC, shared_followers DESC LIMIT 20;
SELECT * FROM broadcaster_top_neighbors LIMIT 20;
SELECT * FROM top10_broadcaster_recommendations LIMIT 100;

-- Optional event-history reconstruction template (edit source names/columns):
-- CREATE OR REPLACE TEMP VIEW follow_edges AS
-- SELECT CAST(user_id AS STRING) AS user_id,
--        CAST(broadcaster_id AS STRING) AS broadcaster_id
-- FROM (
--   SELECT *, ROW_NUMBER() OVER (
--     PARTITION BY user_id, broadcaster_id
--     ORDER BY event_timestamp DESC, event_id DESC
--   ) AS rn
--   FROM analytics.follow_events
--   WHERE user_id IS NOT NULL AND broadcaster_id IS NOT NULL
-- ) latest
-- WHERE rn = 1 AND action = 'follow';
-- CACHE TABLE follow_edges;
-- Place this BEFORE section 2, replacing section 1, if events are available.
