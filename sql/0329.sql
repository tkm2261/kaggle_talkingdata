-- talking4.mst_dayiphourapp
SELECT
  day, hour, ip, count(1) cnt
FROM `talking2.train_test`
GROUP BY day, hour, ip

-- talking4.mst_dayipchannel
SELECT
  day, ip, channel, count(1) cnt
FROM `talking2.train_test`
GROUP BY day, ip, channel

-- talking4.mst_dayipdist
SELECT
  day, ip,
  count(distinct app) app,
  count(distinct device) device,
  count(distinct channel) channel,
  count(distinct os) os
FROM `talking2.train_test`
GROUP BY day, ip


-- takling4.train_test2
SELECT
  t.click_id,
  t.is_attributed ,
  t.day,
  t.span,
  t.hour,
  t.ip,
  t.os,
  t.app,
  t.channel,
  t.device,
  TIMESTAMP_DIFF(click_time, LAG(click_time, 1) OVER(partition by t.day, t.ip, t.os, t.device  order by click_time), SECOND) as click_diff_1,
  TIMESTAMP_DIFF(click_time, LAG(click_time, 2) OVER(partition by t.day, t.ip, t.os, t.device order by click_time), SECOND) as click_diff_2,
  TIMESTAMP_DIFF(click_time, LAG(click_time, 3) OVER(partition by t.day, t.ip, t.os, t.device order by click_time), SECOND) as click_diff_3,
  TIMESTAMP_DIFF(click_time, LAG(click_time, 4) OVER(partition by t.day, t.ip, t.os, t.device order by click_time), SECOND) as click_diff_4,
  TIMESTAMP_DIFF(click_time, LAG(click_time, 5) OVER(partition by t.day, t.ip, t.os, t.device order by click_time), SECOND) as click_diff_5,
  AVG(t.is_attributed) OVER(partition by t.ip, t.day, t.hour, t.channel order by click_time ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) as avg_ipdayhour,
  ROW_NUMBER() OVER(partition by t.ip, t.day, t.device, t.os order by click_time) as cnt_ip,
  ROW_NUMBER() OVER(partition by t.ip, t.day, t.channel order by click_time) as cnt_ch,
  a.cnt as cnt_dayiphourapp,
  b.cnt as cnt_dayipchannel,
  c.app as dist_app,
  c.device as dist_device,
  c.channel as dist_channel,
  c.os as dist_os
FROM
  `talking2.train_test` as t
LEFT OUTER JOIN `talking4.mst_dayiphourapp` as a
ON a.day = t.day and a.hour = t.hour and a.ip = t.ip
LEFT OUTER JOIN `talking4.mst_dayipchannel` as b
ON b.day = t.day and b.channel = t.hour and b.ip = t.ip
LEFT OUTER JOIN `talking4.mst_dayipdist` as c
ON c.day = t.day and c.ip = t.ip

SELECT
*
FROM
`talking4.train_test2`
WHERE
  day = 8 AND (click_id is null or click_id >= 0)
