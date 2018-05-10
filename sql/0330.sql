-- talking4.mst_dayiphourapp
SELECT
  day, hour, ip, count(1) cnt,
  TIMESTAMP_DIFF(MAX(click_time), MIN(click_time), SECOND) as diff
FROM `talking2.train_test`
GROUP BY day, hour, ip

-- talking4.mst_dayhouripchannel
SELECT
  day, hour, ip, channel, count(1) cnt,
  TIMESTAMP_DIFF(MAX(click_time), MIN(click_time), SECOND) as diff
FROM `talking2.train_test`
GROUP BY day, hour, ip, channel

-- talking4.mst_dayhouripapp
SELECT
  day, hour, ip, app, count(1) cnt,
  TIMESTAMP_DIFF(MAX(click_time), MIN(click_time), SECOND) as diff
FROM `talking2.train_test`
GROUP BY day, hour, ip, app

-- talking4.mst_dayhouripappchannel
SELECT
  day, hour, ip, app, channel, count(1) cnt,
  TIMESTAMP_DIFF(MAX(click_time), MIN(click_time), SECOND) as diff
FROM `talking2.train_test`
GROUP BY day, hour, ip, app, channel

-- talking4.mst_dayhouripos
SELECT
  day, hour, ip, os, count(1) cnt,
  TIMESTAMP_DIFF(MAX(click_time), MIN(click_time), SECOND) as diff
FROM `talking2.train_test`
GROUP BY day, hour, ip, os

-- talking4.mst_dayhouripdevice
SELECT
  day, hour, ip, device, count(1) cnt,
  TIMESTAMP_DIFF(MAX(click_time), MIN(click_time), SECOND) as diff
FROM `talking2.train_test`
GROUP BY day, hour, ip, device

-- talking4.mst_all
SELECT
  day, hour, ip, os, app, channel, device, count(1) cnt,
  TIMESTAMP_DIFF(MAX(click_time), MIN(click_time), SECOND) as diff
FROM `talking2.train_test`
GROUP BY day, hour, ip, os, app, channel, device


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
  ROW_NUMBER() OVER(partition by t.ip, t.day, t.hour, t.device, t.os order by click_time) as cnt_ip,
  ROW_NUMBER() OVER(partition by t.ip, t.day, t.hour, t.channel order by click_time) as cnt_ch,
  ROW_NUMBER() OVER(partition by t.ip, t.day, t.hour, t.channel, t.app order by click_time) as cnt_ch_app,
  ROW_NUMBER() OVER(partition by t.ip, t.day, t.hour, t.app order by click_time) as cnt_app,
  ROW_NUMBER() OVER(partition by t.ip, t.day, t.hour, t.os order by click_time) as cnt_os,
  ROW_NUMBER() OVER(partition by t.ip, t.day, t.hour, t.device, t.os order by click_time desc) as cnt_ip_r,
  ROW_NUMBER() OVER(partition by t.ip, t.day, t.hour, t.channel order by click_time desc) as cnt_ch_r,
  ROW_NUMBER() OVER(partition by t.ip, t.day, t.hour, t.channel, t.app order by click_time desc) as cnt_ch_app_r,
  ROW_NUMBER() OVER(partition by t.ip, t.day, t.hour, t.app order by click_time desc) as cnt_app_r,
  ROW_NUMBER() OVER(partition by t.ip, t.day, t.hour, t.os order by click_time desc) as cnt_os_r,
  a.cnt as cnt_dayiphourapp,
  b.cnt as cnt_dayhouripchannel,
  c.cnt as cnt_dayhouripapp,
  d.cnt as cnt_dayhouripos,
  e.cnt as cnt_dayhouripappchannel,
  f.cnt as cnt_dayhouripdevice,
  b.cnt / a.cnt as rt_dayhouripchannel,
  c.cnt / a.cnt as rt_dayhouripapp,
  d.cnt / a.cnt as rt_dayhouripos,
  e.cnt / a.cnt as rt_dayhouripappchannel,
  f.cnt / a.cnt as rt_dayhouripdevice,
  g.cnt as cnt_all,
  a.diff as diff_dayiphourapp,
  b.diff as diff_dayhouripchannel,
  c.diff as diff_dayhouripapp,
  d.diff as diff_dayhouripos,
  e.diff as diff_dayhouripappchannel,
  f.diff as diff_dayhouripdevice,
  g.diff as diff_all
FROM
  `talking2.train_test` as t
LEFT OUTER JOIN `talking4.mst_dayiphourapp` as a
ON a.day = t.day and a.hour = t.hour and a.ip = t.ip
LEFT OUTER JOIN `talking4.mst_dayhouripchannel` as b
ON b.day = t.day and b.hour = t.hour and b.channel = t.channel and b.ip = t.ip
LEFT OUTER JOIN `talking4.mst_dayhouripapp` as c
ON c.day = t.day and c.hour = t.hour and c.ip = t.ip and c.app = t.app
LEFT OUTER JOIN `talking4.mst_dayhouripos` as d
ON d.day = t.day and d.hour = t.hour and d.ip = t.ip and d.os = t.os
LEFT OUTER JOIN `talking4.mst_dayhouripappchannel` as e
ON e.day = t.day and e.hour = t.hour and e.ip = t.ip and e.app = t.app and e.channel = t.channel
LEFT OUTER JOIN `talking4.mst_dayhouripdevice` as f
ON f.day = t.day and f.hour = t.hour and f.ip = t.ip and f.device = t.device
LEFT OUTER JOIN `talking4.mst_all` as g
ON g.day = t.day and g.hour = t.hour and g.ip = t.ip
and g.os = t.os and g.app = t.app and g.channel = t.channel and g.device = t.device


SELECT
*
FROM
`talking4.train_test2`
WHERE
  day = 8 AND (click_id is null or click_id >= 0)
