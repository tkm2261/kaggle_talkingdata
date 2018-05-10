SELECT
  count(1),
  count(distinct t.click_id)
FROM
  `talking.old_test2` as o
LEFT OUTER JOIN
  `talking.test2` as t
ON
  o.ip = t.ip AND
  o.app = t.app AND
  o.device = t.device AND
  o.os = t.os AND
  o.channel = t.channel AND
  o.click_time = t.click_time
--  	16538595486	18790469
-- 	78236845	18790469
--	58175885	18790469


-- talking2.old_test2_gb
SELECT
ip, app, device, os, channel, click_time
FROM
  `talking.old_test2` as o
GROUP BY
ip, app, device, os, channel, click_time

-- talking2.all_test2
SELECT
  CASE WHEN t.click_id is null THEN -1 ELSE t.click_id END as click_id,
  o.*,
  TIMESTAMP_DIFF(o.click_time, LAG(o.click_time) OVER(partition by o.ip order by o.click_time), SECOND) as timediff,
  EXTRACT(year from o.click_time) as year,
  EXTRACT(month from o.click_time) as month,
  EXTRACT(day from o.click_time) as day,
  EXTRACT(DAYOFWEEK from o.click_time) as dayofweek,
  EXTRACT(HOUR from o.click_time) as hour,
  EXTRACT(MINUTE from o.click_time) as minute,
  EXTRACT(SECOND from o.click_time) as second
FROM
  `talking2.old_test2_gb` as o
LEFT OUTER JOIN
  `talking.test2` as t
ON
  o.ip = t.ip AND
  o.app = t.app AND
  o.device = t.device AND
  o.os = t.os AND
  o.channel = t.channel AND
  o.click_time = t.click_time

-- takling2.train_test
SELECT
null as click_id,
CASE WHEN hour >= 4 and hour <= 6 THEN 1
     WHEN hour >= 9 and hour <= 11 THEN 2
     WHEN hour >= 13 and hour <= 15 THEN 3 ELSE -1 END span,
ip, app, device, os, channel, click_time, attributed_time, is_attributed, timediff, year, month, day, dayofweek, hour, minute, second
FROM
`talking.train2`
UNION ALL
SELECT
click_id,
CASE WHEN hour >= 4 and hour <= 6 THEN 1
     WHEN hour >= 9 and hour <= 11 THEN 2
     WHEN hour >= 13 and hour <= 15 THEN 3 ELSE -1 END span,
ip, app, device, os, channel, click_time, null as attributed_time, null as is_attributed, timediff, year, month, day, dayofweek, hour, minute, second
FROM
`talking2.all_test2`

-- takling2.train_test2
SELECT
  *,
  TIMESTAMP_DIFF(click_time, LAG(click_time, 1) OVER(partition by ip order by click_time), SECOND) as click_diff_1,
  TIMESTAMP_DIFF(click_time, LAG(click_time, 2) OVER(partition by ip order by click_time), SECOND) as click_diff_2,
  TIMESTAMP_DIFF(click_time, LAG(click_time, 3) OVER(partition by ip order by click_time), SECOND) as click_diff_3,
  TIMESTAMP_DIFF(click_time, LAG(click_time, 4) OVER(partition by ip order by click_time), SECOND) as click_diff_4,
  TIMESTAMP_DIFF(click_time, LAG(click_time, 5) OVER(partition by ip order by click_time), SECOND) as click_diff_5,
  AVG(t.is_attributed) OVER(partition by t.ip, t.day, t.hour order by click_time ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) as avg_ipdayhour,
  ROW_NUMBER() OVER(partition by t.ip order by click_time) as cnt_ip,
  SUM(t.is_attributed) OVER(partition by t.ip order by click_time ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) as sum_ip,
  avg(is_attributed) OVER(partition by ip order by click_time ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) as avg_ip
FROM
  `talking2.train_test` as t
#####


-- takling2.mst_app
SELECT
  8 as day, app, span, avg(is_attributed) avg_app, count(1) cnt_app
FROM `talking.train_test`
WHERE click_time < '2017-11-08 00:00:00' AND click_time >= '2017-11-08 00:00:00'
GROUP BY app, span
UNION ALL
SELECT
  9 as day, app, span, avg(is_attributed) avg_app, count(1) / 2 cnt_app
FROM `talking.train_test`
WHERE click_time < '2017-11-09 00:00:00' AND click_time >= '2017-11-08 00:00:00'
GROUP BY app, span
UNION ALL
SELECT
  10 as day, app, span, avg(is_attributed) avg_app, count(1) / 3 cnt_app
FROM `talking.train_test`
WHERE click_time < '2017-11-10 00:00:00' AND click_time >= '2017-11-08 00:00:00'
GROUP BY app, span

-- takling2.mst_ip
SELECT
  8 as day, ip, span, avg(is_attributed) avg_ip, count(1) cnt_ip
FROM `talking.train_test`
WHERE click_time < '2017-11-08 00:00:00' AND click_time >= '2017-11-08 00:00:00'
GROUP BY ip, span
UNION ALL
SELECT
  9 as day, ip, span, avg(is_attributed) avg_ip, count(1) / 2  cnt_ip
FROM `talking.train_test`
WHERE click_time < '2017-11-09 00:00:00' AND click_time >= '2017-11-08 00:00:00'
GROUP BY ip, span
UNION ALL
SELECT
  10 as day, ip, span, avg(is_attributed) avg_ip, count(1) / 3 cnt_ip
FROM `talking.train_test`
WHERE click_time < '2017-11-10 00:00:00' AND click_time >= '2017-11-08 00:00:00'
GROUP BY ip, span

-- takling2.mst_device
SELECT
  8 as day, device, span, avg(is_attributed) avg_device, count(1) cnt_device
FROM `talking.train_test`
WHERE click_time < '2017-11-08 00:00:00' AND click_time >= '2017-11-08 00:00:00'
GROUP BY device, span
UNION ALL
SELECT
  9 as day, device, span, avg(is_attributed) avg_device, count(1) / 2  cnt_device
FROM `talking.train_test`
WHERE click_time < '2017-11-09 00:00:00' AND click_time >= '2017-11-08 00:00:00'
GROUP BY device, span
UNION ALL
SELECT
  10 as day, device, span, avg(is_attributed) avg_device, count(1) / 3 cnt_device
FROM `talking.train_test`
WHERE click_time < '2017-11-10 00:00:00' AND click_time >= '2017-11-08 00:00:00'
GROUP BY device, span

-- takling2.mst_os
SELECT
  8 as day, os, span, avg(is_attributed) avg_os, count(1) cnt_os
FROM `talking.train_test`
WHERE click_time < '2017-11-08 00:00:00' AND click_time >= '2017-11-08 00:00:00'
GROUP BY os, span
UNION ALL
SELECT
  9 as day, os, span, avg(is_attributed) avg_os, count(1) / 2  cnt_os
FROM `talking.train_test`
WHERE click_time < '2017-11-09 00:00:00' AND click_time >= '2017-11-08 00:00:00'
GROUP BY os, span
UNION ALL
SELECT
  10 as day, os, span, avg(is_attributed) avg_os, count(1) / 3 cnt_os
FROM `talking.train_test`
WHERE click_time < '2017-11-10 00:00:00' AND click_time >= '2017-11-08 00:00:00'
GROUP BY os, span


-- takling2.mst_ch
SELECT
  8 as day, channel, span, avg(is_attributed) avg_channel, count(1) cnt_channel
FROM `talking.train_test`
WHERE click_time < '2017-11-08 00:00:00' AND click_time >= '2017-11-08 00:00:00'
GROUP BY channel, span
UNION ALL
SELECT
  9 as day, channel, span, avg(is_attributed) avg_channel, count(1) / 2 cnt_channel
FROM `talking.train_test`
WHERE click_time < '2017-11-09 00:00:00' AND click_time >= '2017-11-08 00:00:00'
GROUP BY channel, span
UNION ALL
SELECT
  10 as day, channel, span, avg(is_attributed) avg_channel, count(1) / 3 cnt_channel
FROM `talking.train_test`
WHERE click_time < '2017-11-10 00:00:00' AND click_time >= '2017-11-08 00:00:00'
GROUP BY channel, span


-- takling2.mst_hour
SELECT
  8 as day, hour, avg(is_attributed) avg_hour, count(1) cnt_hour
FROM `talking.train_test`
WHERE click_time < '2017-11-08 00:00:00' AND click_time >= '2017-11-08 00:00:00'
GROUP BY hour
UNION ALL
SELECT
  9 as day, hour, avg(is_attributed) avg_hour, count(1) / 2 cnt_hour
FROM `talking.train_test`
WHERE click_time < '2017-11-09 00:00:00' AND click_time >= '2017-11-08 00:00:00'
GROUP BY hour
UNION ALL
SELECT
  10 as day, hour, avg(is_attributed) avg_hour, count(1) / 3 cnt_hour
FROM `talking.train_test`
WHERE click_time < '2017-11-10 00:00:00' AND click_time >= '2017-11-08 00:00:00'
GROUP BY hour


-- takling2.mst_minute
SELECT
  8 as day, minute, avg(is_attributed) avg_minute, count(1) cnt_minute
FROM `talking.train_test`
WHERE click_time < '2017-11-08 00:00:00' AND click_time >= '2017-11-08 00:00:00'
GROUP BY minute
UNION ALL
SELECT
  9 as day, minute, avg(is_attributed) avg_minute, count(1) / 2 cnt_minute
FROM `talking.train_test`
WHERE click_time < '2017-11-09 00:00:00' AND click_time >= '2017-11-08 00:00:00'
GROUP BY minute
UNION ALL
SELECT
  10 as day, minute, avg(is_attributed) avg_minute, count(1) /3 cnt_minute
FROM `talking.train_test`
WHERE click_time < '2017-11-10 00:00:00' AND click_time >= '2017-11-08 00:00:00'
GROUP BY minute


-- takling2.mst_second
SELECT
  8 as day, second, avg(is_attributed) avg_second, count(1) cnt_second
FROM `talking.train_test`
WHERE click_time < '2017-11-08 00:00:00' AND click_time >= '2017-11-08 00:00:00'
GROUP BY second
UNION ALL
SELECT
  9 as day, second, avg(is_attributed) avg_second, count(1) / 2 cnt_second
FROM `talking.train_test`
WHERE click_time < '2017-11-09 00:00:00' AND click_time >= '2017-11-08 00:00:00'
GROUP BY second
UNION ALL
SELECT
  10 as day, second, avg(is_attributed) avg_second, count(1) / 3 cnt_second
FROM `talking.train_test`
WHERE click_time < '2017-11-10 00:00:00' AND click_time >= '2017-11-08 00:00:00'
GROUP BY second


#####

-- takling2.mst_app_day
SELECT
  8 as day, app,  avg(is_attributed) avg_app, count(1) cnt_app
FROM `talking.train_test`
WHERE click_time < '2017-11-08 00:00:00' AND click_time >= '2017-11-08 00:00:00'
GROUP BY app
UNION ALL
SELECT
  9 as day, app,  avg(is_attributed) avg_app, count(1) / 2 cnt_app
FROM `talking.train_test`
WHERE click_time < '2017-11-09 00:00:00' AND click_time >= '2017-11-08 00:00:00'
GROUP BY app
UNION ALL
SELECT
  10 as day, app,  avg(is_attributed) avg_app, count(1) / 3 cnt_app
FROM `talking.train_test`
WHERE click_time < '2017-11-10 00:00:00' AND click_time >= '2017-11-08 00:00:00'
GROUP BY app

-- takling2.mst_ip_day
SELECT
  8 as day, ip,  avg(is_attributed) avg_ip, count(1) cnt_ip
FROM `talking.train_test`
WHERE click_time < '2017-11-08 00:00:00' AND click_time >= '2017-11-08 00:00:00'
GROUP BY ip
UNION ALL
SELECT
  9 as day, ip,  avg(is_attributed) avg_ip, count(1) / 2 cnt_ip
FROM `talking.train_test`
WHERE click_time < '2017-11-09 00:00:00' AND click_time >= '2017-11-08 00:00:00'
GROUP BY ip
UNION ALL
SELECT
  10 as day, ip,  avg(is_attributed) avg_ip, count(1) / 3 cnt_ip
FROM `talking.train_test`
WHERE click_time < '2017-11-10 00:00:00' AND click_time >= '2017-11-08 00:00:00'
GROUP BY ip

-- takling2.mst_device_day
SELECT
  8 as day, device,  avg(is_attributed) avg_device, count(1) cnt_device
FROM `talking.train_test`
WHERE click_time < '2017-11-08 00:00:00' AND click_time >= '2017-11-08 00:00:00'
GROUP BY device
UNION ALL
SELECT
  9 as day, device,  avg(is_attributed) avg_device, count(1) / 2 cnt_device
FROM `talking.train_test`
WHERE click_time < '2017-11-09 00:00:00' AND click_time >= '2017-11-08 00:00:00'
GROUP BY device
UNION ALL
SELECT
  10 as day, device,  avg(is_attributed) avg_device, count(1) / 3 cnt_device
FROM `talking.train_test`
WHERE click_time < '2017-11-10 00:00:00' AND click_time >= '2017-11-08 00:00:00'
GROUP BY device

-- takling2.mst_os_day
SELECT
  8 as day, os,  avg(is_attributed) avg_os, count(1) cnt_os
FROM `talking.train_test`
WHERE click_time < '2017-11-08 00:00:00' AND click_time >= '2017-11-08 00:00:00'
GROUP BY os
UNION ALL
SELECT
  9 as day, os,  avg(is_attributed) avg_os, count(1) / 2 cnt_os
FROM `talking.train_test`
WHERE click_time < '2017-11-09 00:00:00' AND click_time >= '2017-11-08 00:00:00'
GROUP BY os
UNION ALL
SELECT
  10 as day, os,  avg(is_attributed) avg_os, count(1) / 3 cnt_os
FROM `talking.train_test`
WHERE click_time < '2017-11-10 00:00:00' AND click_time >= '2017-11-08 00:00:00'
GROUP BY os


-- takling2.mst_ch_day
SELECT
  8 as day, channel,  avg(is_attributed) avg_channel, count(1) cnt_channel
FROM `talking.train_test`
WHERE click_time < '2017-11-08 00:00:00' AND click_time >= '2017-11-08 00:00:00'
GROUP BY channel
UNION ALL
SELECT
  9 as day, channel,  avg(is_attributed) avg_channel, count(1) / 2 cnt_channel
FROM `talking.train_test`
WHERE click_time < '2017-11-09 00:00:00' AND click_time >= '2017-11-08 00:00:00'
GROUP BY channel
UNION ALL
SELECT
  10 as day, channel,  avg(is_attributed) avg_channel, count(1) / 3 cnt_channel
FROM `talking.train_test`
WHERE click_time < '2017-11-10 00:00:00' AND click_time >= '2017-11-08 00:00:00'
GROUP BY channel

#####




#####


######
-- takling2.train_test3
SELECT
  t.*,
  a.avg_app, d.avg_device, o.avg_os, c.avg_channel, h.avg_hour, i.avg_ip as avg_ip_span, m.avg_minute, s.avg_second,
  aa.avg_app avg_app2, dd.avg_device avg_device2, oo.avg_os avg_os2, cc.avg_channel avg_channel2, hh.avg_hour avg_hour2, ii.avg_ip as avg_ip_span2,
  a.cnt_app, d.cnt_device, o.cnt_os, c.cnt_channel, h.cnt_hour, i.cnt_ip as cnt_ip_span, m.cnt_minute, s.cnt_second,
  aa.cnt_app cnt_app2, dd.cnt_device cnt_device2, oo.cnt_os cnt_os2, cc.cnt_channel cnt_channel2, hh.cnt_hour cnt_hour2, ii.cnt_ip as cnt_ip_span2
FROM `talking2.train_test2` as t
LEFT OUTER JOIN talking.mst_app as a
ON a.app = t.app AND a.span = t.span AND a.day = t.day
LEFT OUTER JOIN talking.mst_device as d
ON d.device = t.device AND d.span = t.span AND d.day = t.day
LEFT OUTER JOIN talking.mst_os as o
ON o.os = t.os AND o.span = t.span AND o.day = t.day
LEFT OUTER JOIN talking.mst_ch as c
ON c.channel = t.channel AND c.span = t.span AND c.day = t.day
LEFT OUTER JOIN talking.mst_ip as i
ON i.ip = t.ip AND i.span = t.span AND i.day = t.day
LEFT OUTER JOIN talking.mst_hour as h
ON h.hour = t.hour AND h.day = t.day
LEFT OUTER JOIN talking.mst_minute as m
ON m.minute = t.minute AND m.day = t.day
LEFT OUTER JOIN talking.mst_second as s
ON s.second = t.second AND s.day = t.day
LEFT OUTER JOIN talking.mst_app_day as aa
ON aa.app = t.app AND aa.day = t.day
LEFT OUTER JOIN talking.mst_device_day as dd
ON dd.device = t.device AND dd.day = t.day
LEFT OUTER JOIN talking.mst_os_day as oo
ON oo.os = t.os AND oo.day = t.day
LEFT OUTER JOIN talking.mst_ch_day as cc
ON cc.channel = t.channel AND cc.day = t.day
LEFT OUTER JOIN talking.mst_ip_day as ii
ON ii.ip = t.ip AND ii.day = t.day
LEFT OUTER JOIN talking.mst_hour_day as hh
ON hh.hour = t.hour AND hh.day = t.day


-- dmt_train_span_all
SELECT
click_id,
ip,
app,
click_diff_1, click_diff_2, click_diff_3, click_diff_4, click_diff_5,
avg_ipdayhour,
LAG(avg_ipdayhour, 1) OVER(partition by ip order by click_time) avg_ipdayhour_1,
LAG(avg_ipdayhour, 2) OVER(partition by ip order by click_time) avg_ipdayhour_2,
LAG(avg_ipdayhour, 3) OVER(partition by ip order by click_time) avg_ipdayhour_3,
LAG(avg_ipdayhour, 4) OVER(partition by ip order by click_time) avg_ipdayhour_4,
avg_ip,
LAG(avg_ip, 1) OVER(partition by ip order by click_time) avg_ip_1,
LAG(avg_ip, 2) OVER(partition by ip order by click_time) avg_ip_2,
LAG(avg_ip, 3) OVER(partition by ip order by click_time) avg_ip_3,
LAG(avg_ip, 4) OVER(partition by ip order by click_time) avg_ip_4,
LAG(app, 1) OVER(partition by ip order by click_time) app_1,
LAG(app, 2) OVER(partition by ip order by click_time) app_2,
LAG(app, 3) OVER(partition by ip order by click_time) app_3,
LAG(app, 4) OVER(partition by ip order by click_time) app_4,
device,
LAG(device, 1) OVER(partition by ip order by click_time) device_1,
LAG(device, 2) OVER(partition by ip order by click_time) device_2,
LAG(device, 3) OVER(partition by ip order by click_time) device_3,
LAG(device, 4) OVER(partition by ip order by click_time) device_4,
os,
LAG(os, 1) OVER(partition by ip order by click_time) os_1,
LAG(os, 2) OVER(partition by ip order by click_time) os_2,
LAG(os, 3) OVER(partition by ip order by click_time) os_3,
LAG(os, 4) OVER(partition by ip order by click_time) os_4,
channel,
LAG(channel, 1) OVER(partition by ip order by click_time) channel_1,
LAG(channel, 2) OVER(partition by ip order by click_time) channel_2,
LAG(channel, 3) OVER(partition by ip order by click_time) channel_3,
LAG(channel, 4) OVER(partition by ip order by click_time) channel_4,
is_attributed,
LAG(is_attributed, 1) OVER(partition by ip order by click_time) is_attributed_1,
LAG(is_attributed, 2) OVER(partition by ip order by click_time) is_attributed_2,
LAG(is_attributed, 3) OVER(partition by ip order by click_time) is_attributed_3,
LAG(is_attributed, 4) OVER(partition by ip order by click_time) is_attributed_4,
LAG(is_attributed, 5) OVER(partition by ip order by click_time) is_attributed_5,
hour,
LAG(hour, 1) OVER(partition by ip order by click_time) hour_1,
LAG(hour, 2) OVER(partition by ip order by click_time) hour_2,
LAG(hour, 3) OVER(partition by ip order by click_time) hour_3,
LAG(hour, 4) OVER(partition by ip order by click_time) hour_4,
minute,
LAG(minute, 1) OVER(partition by ip order by click_time) minute_1,
LAG(minute, 2) OVER(partition by ip order by click_time) minute_2,
LAG(minute, 3) OVER(partition by ip order by click_time) minute_3,
LAG(minute, 4) OVER(partition by ip order by click_time) minute_4,
second,
LAG(second, 1) OVER(partition by ip order by click_time) second_1,
LAG(second, 2) OVER(partition by ip order by click_time) second_2,
LAG(second, 3) OVER(partition by ip order by click_time) second_3,
LAG(second, 4) OVER(partition by ip order by click_time) second_4,
avg_app,
LAG(avg_app, 1) OVER(partition by ip order by click_time) avg_app_1,
LAG(avg_app, 2) OVER(partition by ip order by click_time) avg_app_2,
LAG(avg_app, 3) OVER(partition by ip order by click_time) avg_app_3,
LAG(avg_app, 4) OVER(partition by ip order by click_time) avg_app_4,
avg_device,
LAG(avg_device, 1) OVER(partition by ip order by click_time) avg_device_1,
LAG(avg_device, 2) OVER(partition by ip order by click_time) avg_device_2,
LAG(avg_device, 3) OVER(partition by ip order by click_time) avg_device_3,
LAG(avg_device, 4) OVER(partition by ip order by click_time) avg_device_4,
avg_os,
LAG(avg_os, 1) OVER(partition by ip order by click_time) avg_os_1,
LAG(avg_os, 2) OVER(partition by ip order by click_time) avg_os_2,
LAG(avg_os, 3) OVER(partition by ip order by click_time) avg_os_3,
LAG(avg_os, 4) OVER(partition by ip order by click_time) avg_os_4,
avg_channel,
LAG(avg_channel, 1) OVER(partition by ip order by click_time) avg_channel_1,
LAG(avg_channel, 2) OVER(partition by ip order by click_time) avg_channel_2,
LAG(avg_channel, 3) OVER(partition by ip order by click_time) avg_channel_3,
LAG(avg_channel, 4) OVER(partition by ip order by click_time) avg_channel_4,
avg_hour,
LAG(avg_hour, 1) OVER(partition by ip order by click_time) avg_hour_1,
LAG(avg_hour, 2) OVER(partition by ip order by click_time) avg_hour_2,
LAG(avg_hour, 3) OVER(partition by ip order by click_time) avg_hour_3,
LAG(avg_hour, 4) OVER(partition by ip order by click_time) avg_hour_4,
avg_ip_span,
LAG(avg_ip_span, 1) OVER(partition by ip order by click_time) avg_ip_span_1,
LAG(avg_ip_span, 2) OVER(partition by ip order by click_time) avg_ip_span_2,
LAG(avg_ip_span, 3) OVER(partition by ip order by click_time) avg_ip_span_3,
LAG(avg_ip_span, 4) OVER(partition by ip order by click_time) avg_ip_span_4,
avg_minute,
LAG(avg_minute, 1) OVER(partition by ip order by click_time) avg_minute_1,
LAG(avg_minute, 2) OVER(partition by ip order by click_time) avg_minute_2,
LAG(avg_minute, 3) OVER(partition by ip order by click_time) avg_minute_3,
LAG(avg_minute, 4) OVER(partition by ip order by click_time) avg_minute_4,
avg_second,
LAG(avg_second, 1) OVER(partition by ip order by click_time) avg_second_1,
LAG(avg_second, 2) OVER(partition by ip order by click_time) avg_second_2,
LAG(avg_second, 3) OVER(partition by ip order by click_time) avg_second_3,
LAG(avg_second, 4) OVER(partition by ip order by click_time) avg_second_4,
avg_app2,
LAG(avg_app2, 1) OVER(partition by ip order by click_time) avg_app2_1,
LAG(avg_app2, 2) OVER(partition by ip order by click_time) avg_app2_2,
LAG(avg_app2, 3) OVER(partition by ip order by click_time) avg_app2_3,
LAG(avg_app2, 4) OVER(partition by ip order by click_time) avg_app2_4,
avg_device2,
LAG(avg_device2, 1) OVER(partition by ip order by click_time) avg_device2_1,
LAG(avg_device2, 2) OVER(partition by ip order by click_time) avg_device2_2,
LAG(avg_device2, 3) OVER(partition by ip order by click_time) avg_device2_3,
LAG(avg_device2, 4) OVER(partition by ip order by click_time) avg_device2_4,
avg_os2,
LAG(avg_os2, 1) OVER(partition by ip order by click_time) avg_os2_1,
LAG(avg_os2, 2) OVER(partition by ip order by click_time) avg_os2_2,
LAG(avg_os2, 3) OVER(partition by ip order by click_time) avg_os2_3,
LAG(avg_os2, 4) OVER(partition by ip order by click_time) avg_os2_4,
avg_channel2,
LAG(avg_channel2, 1) OVER(partition by ip order by click_time) avg_channel2_1,
LAG(avg_channel2, 2) OVER(partition by ip order by click_time) avg_channel2_2,
LAG(avg_channel2, 3) OVER(partition by ip order by click_time) avg_channel2_3,
LAG(avg_channel2, 4) OVER(partition by ip order by click_time) avg_channel2_4,
avg_ip_span2,
LAG(avg_ip_span2, 1) OVER(partition by ip order by click_time) avg_ip_span2_1,
LAG(avg_ip_span2, 2) OVER(partition by ip order by click_time) avg_ip_span2_2,
LAG(avg_ip_span2, 3) OVER(partition by ip order by click_time) avg_ip_span2_3,
LAG(avg_ip_span2, 4) OVER(partition by ip order by click_time) avg_ip_span2_4,
cnt_app,
LAG(cnt_app, 1) OVER(partition by ip order by click_time) cnt_app_1,
LAG(cnt_app, 2) OVER(partition by ip order by click_time) cnt_app_2,
LAG(cnt_app, 3) OVER(partition by ip order by click_time) cnt_app_3,
LAG(cnt_app, 4) OVER(partition by ip order by click_time) cnt_app_4,
cnt_device,
LAG(cnt_device, 1) OVER(partition by ip order by click_time) cnt_device_1,
LAG(cnt_device, 2) OVER(partition by ip order by click_time) cnt_device_2,
LAG(cnt_device, 3) OVER(partition by ip order by click_time) cnt_device_3,
LAG(cnt_device, 4) OVER(partition by ip order by click_time) cnt_device_4,
cnt_os,
LAG(cnt_os, 1) OVER(partition by ip order by click_time) cnt_os_1,
LAG(cnt_os, 2) OVER(partition by ip order by click_time) cnt_os_2,
LAG(cnt_os, 3) OVER(partition by ip order by click_time) cnt_os_3,
LAG(cnt_os, 4) OVER(partition by ip order by click_time) cnt_os_4,
cnt_channel,
LAG(cnt_channel, 1) OVER(partition by ip order by click_time) cnt_channel_1,
LAG(cnt_channel, 2) OVER(partition by ip order by click_time) cnt_channel_2,
LAG(cnt_channel, 3) OVER(partition by ip order by click_time) cnt_channel_3,
LAG(cnt_channel, 4) OVER(partition by ip order by click_time) cnt_channel_4,
cnt_hour,
LAG(cnt_hour, 1) OVER(partition by ip order by click_time) cnt_hour_1,
LAG(cnt_hour, 2) OVER(partition by ip order by click_time) cnt_hour_2,
LAG(cnt_hour, 3) OVER(partition by ip order by click_time) cnt_hour_3,
LAG(cnt_hour, 4) OVER(partition by ip order by click_time) cnt_hour_4,
cnt_ip_span,
LAG(cnt_ip_span, 1) OVER(partition by ip order by click_time) cnt_ip_span_1,
LAG(cnt_ip_span, 2) OVER(partition by ip order by click_time) cnt_ip_span_2,
LAG(cnt_ip_span, 3) OVER(partition by ip order by click_time) cnt_ip_span_3,
LAG(cnt_ip_span, 4) OVER(partition by ip order by click_time) cnt_ip_span_4,
cnt_minute,
LAG(cnt_minute, 1) OVER(partition by ip order by click_time) cnt_minute_1,
LAG(cnt_minute, 2) OVER(partition by ip order by click_time) cnt_minute_2,
LAG(cnt_minute, 3) OVER(partition by ip order by click_time) cnt_minute_3,
LAG(cnt_minute, 4) OVER(partition by ip order by click_time) cnt_minute_4,
cnt_second,
LAG(cnt_second, 1) OVER(partition by ip order by click_time) cnt_second_1,
LAG(cnt_second, 2) OVER(partition by ip order by click_time) cnt_second_2,
LAG(cnt_second, 3) OVER(partition by ip order by click_time) cnt_second_3,
LAG(cnt_second, 4) OVER(partition by ip order by click_time) cnt_second_4,
cnt_app2,
LAG(cnt_app2, 1) OVER(partition by ip order by click_time) cnt_app2_1,
LAG(cnt_app2, 2) OVER(partition by ip order by click_time) cnt_app2_2,
LAG(cnt_app2, 3) OVER(partition by ip order by click_time) cnt_app2_3,
LAG(cnt_app2, 4) OVER(partition by ip order by click_time) cnt_app2_4,
cnt_device2,
LAG(cnt_device2, 1) OVER(partition by ip order by click_time) cnt_device2_1,
LAG(cnt_device2, 2) OVER(partition by ip order by click_time) cnt_device2_2,
LAG(cnt_device2, 3) OVER(partition by ip order by click_time) cnt_device2_3,
LAG(cnt_device2, 4) OVER(partition by ip order by click_time) cnt_device2_4,
cnt_os2,
LAG(cnt_os2, 1) OVER(partition by ip order by click_time) cnt_os2_1,
LAG(cnt_os2, 2) OVER(partition by ip order by click_time) cnt_os2_2,
LAG(cnt_os2, 3) OVER(partition by ip order by click_time) cnt_os2_3,
LAG(cnt_os2, 4) OVER(partition by ip order by click_time) cnt_os2_4,
cnt_channel2,
LAG(cnt_channel2, 1) OVER(partition by ip order by click_time) cnt_channel2_1,
LAG(cnt_channel2, 2) OVER(partition by ip order by click_time) cnt_channel2_2,
LAG(cnt_channel2, 3) OVER(partition by ip order by click_time) cnt_channel2_3,
LAG(cnt_channel2, 4) OVER(partition by ip order by click_time) cnt_channel2_4,
cnt_ip_span2,
LAG(cnt_ip_span2, 1) OVER(partition by ip order by click_time) cnt_ip_span2_1,
LAG(cnt_ip_span2, 2) OVER(partition by ip order by click_time) cnt_ip_span2_2,
LAG(cnt_ip_span2, 3) OVER(partition by ip order by click_time) cnt_ip_span2_3,
LAG(cnt_ip_span2, 4) OVER(partition by ip order by click_time) cnt_ip_span2_4
FROM
`talking2.train_test3`
WHERE
  day = 8 AND (click_id is null or click_id >= 0)
