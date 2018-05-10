-- takling3.mst_ipdayhour_day
SELECT
  day, ip, hour, avg(is_attributed) avg_app, count(1) cnt_app
FROM `talking2.train_test`
WHERE click_time >= '2017-11-08 04:00:00'
GROUP BY day, ip, hour

-- takling3.mst_ipapp_day
SELECT
  ip, app,  avg(is_attributed) avg_app, count(1) cnt_app
FROM `talking2.train_test`
WHERE click_time >= '2017-11-08 04:00:00'
GROUP BY ip, app

-- takling3.mst_ipappos_day
SELECT
  ip, app, os, avg(is_attributed) avg_app, count(1) cnt_app
FROM `talking2.train_test`
WHERE click_time >= '2017-11-08 04:00:00'
GROUP BY ip, app, os

-- takling3.mst_ipdayhour_day
SELECT
  day, ip, hour,  avg(is_attributed) avg_app, count(1) cnt_app
FROM `talking2.train_test`
WHERE click_time >= '2017-11-08 04:00:00'
GROUP BY day, ip, hour


-- takling3.mst_app
SELECT
  app,  avg(is_attributed) avg_app, count(1) cnt_app
FROM `talking2.train_test`
WHERE click_time >= '2017-11-08 04:00:00'
GROUP BY app

-- takling3.mst_device
SELECT
  device, avg(is_attributed) avg_device, count(1) cnt_device
FROM `talking2.train_test`
WHERE click_time >= '2017-11-08 04:00:00'
GROUP BY device

-- takling3.mst_os
SELECT
  os, avg(is_attributed) avg_os, count(1) cnt_os
FROM `talking2.train_test`
WHERE click_time >= '2017-11-08 04:00:00'
GROUP BY os

-- takling3.mst_channel
SELECT
  channel, avg(is_attributed) avg_channel, count(1) cnt_channel
FROM `talking2.train_test`
WHERE click_time >= '2017-11-08 04:00:00'
GROUP BY channel

-- takling3.ts_ip
SELECT ip
FROM `talking.test`
GROUP BY ip

-- takling3.ts_app
SELECT app
FROM `talking.test`
GROUP BY app

-- takling3.ts_device
SELECT device
FROM `talking.test`
GROUP BY device

-- takling3.ts_os
SELECT os
FROM `talking.test`
GROUP BY os

-- takling3.ts_channel
SELECT channel
FROM `talking.test`
GROUP BY channel

FROM `talking2.train_test3` as t

-- takling3.train_test3
SELECT
t.click_id,
t.click_time,
t.span,
i.ip,
a.app,
d.device,
o.os,
c.channel,
t.is_attributed,
t.day,
t.hour,
t.minute,
t.click_diff_1,
t.click_diff_2,
t.click_diff_3,
t.click_diff_4,
t.click_diff_5,
t.avg_ipdayhour,
t.cnt_ip,
t.sum_ip,
t.avg_ip
FROM `talking2.train_test2` as t
LEFT OUTER JOIN `takling3.ts_ip` as i
ON t.ip = i.ip
LEFT OUTER JOIN `takling3.ts_app` as a
ON t.app = a.app
LEFT OUTER JOIN `takling3.ts_device` as d
ON t.device = d.device
LEFT OUTER JOIN `takling3.ts_os` as o
ON t.os = o.os
LEFT OUTER JOIN `takling3.ts_channel` as c
ON t.channel = c.channel

-- takling3.train_test4
SELECT
t.click_id,
t.click_time,
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
o.avg_app avg_ipappos, o.cnt_app cnt_ipappos,
aa.avg_app avg_app2, aa.cnt_app cnt_app2,
d.avg_device, d.cnt_device,
oo.avg_os, oo.cnt_os,
c.avg_channel, c.cnt_channel
FROM `talking3.train_test3` as t
LEFT OUTER JOIN `talking3.mst_ipdayhour_day` as i
ON t.day = i.day and t.ip = i.ip and t.hour = i.hour
LEFT OUTER JOIN `talking3.mst_ipapp_day` as a
ON t.ip = a.ip and t.app = a.app
LEFT OUTER JOIN `talking3.mst_ipappos_day` as o
ON t.ip = o.ip and t.os = o.os and o.app = t.app
LEFT OUTER JOIN `talking3.mst_app` as aa
ON aa.app = t.app
LEFT OUTER JOIN `talking3.mst_device` as d
ON d.device = t.device
LEFT OUTER JOIN `talking3.mst_os` as oo
ON oo.os = t.os
LEFT OUTER JOIN `talking3.mst_channel` as c
ON c.channel = t.channel

-- dmt_train_kernel2
SELECT
click_id,
ip,
app,
cnt_ip,
sum_ip,
avg_ip,
click_diff_1, click_diff_2, click_diff_3, click_diff_4, click_diff_5,
avg_ipdayhour,
LAG(avg_ipdayhour, 1) OVER(partition by ip order by click_time) avg_ipdayhour_1,
LAG(avg_ipdayhour, 2) OVER(partition by ip order by click_time) avg_ipdayhour_2,
LAG(avg_ipdayhour, 3) OVER(partition by ip order by click_time) avg_ipdayhour_3,
LAG(avg_ipdayhour, 4) OVER(partition by ip order by click_time) avg_ipdayhour_4,
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
cnt_ipdayhour,
LAG(cnt_ipdayhour, 1) OVER(partition by ip order by click_time) cnt_ipdayhour_1,
LAG(cnt_ipdayhour, 2) OVER(partition by ip order by click_time) cnt_ipdayhour_2,
LAG(cnt_ipdayhour, 3) OVER(partition by ip order by click_time) cnt_ipdayhour_3,
LAG(cnt_ipdayhour, 4) OVER(partition by ip order by click_time) cnt_ipdayhour_4,
cnt_ipapp,
LAG(cnt_ipapp, 1) OVER(partition by ip order by click_time) cnt_ipapp_1,
LAG(cnt_ipapp, 2) OVER(partition by ip order by click_time) cnt_ipapp_2,
LAG(cnt_ipapp, 3) OVER(partition by ip order by click_time) cnt_ipapp_3,
LAG(cnt_ipapp, 4) OVER(partition by ip order by click_time) cnt_ipapp_4,
cnt_ipappos,
LAG(cnt_ipappos, 1) OVER(partition by ip order by click_time) cnt_ipappos_1,
LAG(cnt_ipappos, 2) OVER(partition by ip order by click_time) cnt_ipappos_2,
LAG(cnt_ipappos, 3) OVER(partition by ip order by click_time) cnt_ipappos_3,
LAG(cnt_ipappos, 4) OVER(partition by ip order by click_time) cnt_ipappos_4,
avg_ipdayhour2,
avg_ipapp,
avg_ipappos
FROM
`talking3.train_test3`
WHERE
  day = 8 AND (click_id is null or click_id >= 0) AND span > 0
