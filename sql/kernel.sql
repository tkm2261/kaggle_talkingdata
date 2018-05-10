-- takling3.mst_ipdayhour_day
SELECT
  8 as day, ip, hour,  avg(is_attributed) avg_app, count(1) cnt_app
FROM `talking.train_test`
WHERE click_time < '2017-11-08 00:00:00' AND click_time >= '2017-11-07 00:00:00'
GROUP BY ip, hour
UNION ALL
SELECT
  9 as day, ip, hour,  avg(is_attributed) avg_app, count(1) / 2 cnt_app
FROM `talking.train_test`
WHERE click_time < '2017-11-09 00:00:00' AND click_time >= '2017-11-08 00:00:00'
GROUP BY ip, hour
UNION ALL
SELECT
  10 as day, ip, hour,  avg(is_attributed) avg_app, count(1) / 3 cnt_app
FROM `talking.train_test`
WHERE click_time < '2017-11-10 00:00:00' AND click_time >= '2017-11-09 00:00:00'
GROUP BY ip, hour


-- takling3.mst_ipapp_day
SELECT
  8 as day, ip, app,  avg(is_attributed) avg_app, count(1) cnt_app
FROM `talking.train_test`
WHERE click_time < '2017-11-08 00:00:00' AND click_time >= '2017-11-07 00:00:00'
GROUP BY ip, app
UNION ALL
SELECT
  9 as day, ip, app,  avg(is_attributed) avg_app, count(1) cnt_app
FROM `talking.train_test`
WHERE click_time < '2017-11-09 00:00:00' AND click_time >= '2017-11-08 00:00:00'
GROUP BY ip, app
UNION ALL
SELECT
  10 as day, ip, app,  avg(is_attributed) avg_app, count(1) cnt_app
FROM `talking.train_test`
WHERE click_time < '2017-11-10 00:00:00' AND click_time >= '2017-11-09 00:00:00'
GROUP BY ip, app

-- takling3.mst_ipappos_day
SELECT
  8 as day, ip, app,  os, avg(is_attributed) avg_app, count(1) cnt_app
FROM `talking.train_test`
WHERE click_time < '2017-11-08 00:00:00' AND click_time >= '2017-11-07 00:00:00'
GROUP BY ip, app, os
UNION ALL
SELECT
  9 as day, ip, app, os,  avg(is_attributed) avg_app, count(1) cnt_app
FROM `talking.train_test`
WHERE click_time < '2017-11-09 00:00:00' AND click_time >= '2017-11-08 00:00:00'
GROUP BY ip, app, os
UNION ALL
SELECT
  10 as day, ip, app, os, avg(is_attributed) avg_app, count(1) cnt_app
FROM `talking.train_test`
WHERE click_time < '2017-11-10 00:00:00' AND click_time >= '2017-11-09 00:00:00'
GROUP BY ip, app, os

-- takling2.train_test3
SELECT
t.click_id,
t.span,
t.ip,
t.app,
t.device,
t.os,
t.channel,
t.is_attributed,
t.day,
t.hour,
t.minute,
t.second,
t.click_diff_1,
t.click_diff_2,
t.click_diff_3,
t.click_diff_4,
t.click_diff_5,
t.avg_ipdayhour,
t.cnt_ip,
t.sum_ip,
t.avg_ip,
i.avg_app avg_ipdayhour2, i.cnt_app cnt_ipdayhour,
a.avg_app avg_ipapp, a.cnt_app cnt_ipapp,
o.avg_app avg_ipappos, o.cnt_app cnt_ipappos
FROM `talking2.train_test2` as t
LEFT OUTER JOIN `talking3.mst_ipdayhour_day` as i
ON t.day = i.day and t.ip = i.ip and t.hour = i.hour
LEFT OUTER JOIN `talking3.mst_ipapp_day` as a
ON t.day = a.day and t.ip = a.ip and t.app = a.app
LEFT OUTER JOIN `talking3.mst_ipappos_day` as o
ON t.day = o.day and t.ip = o.ip and t.os = o.os and o.app = t.app


SELECT
*
FROM
`talking3.train_test3`
WHERE
  day = 8 AND (click_id is null or click_id >= 0)
