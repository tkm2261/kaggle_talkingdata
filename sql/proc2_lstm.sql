

-- takling.train_test3_8
SELECT
  t.*,
  a.avg_app, d.avg_device, o.avg_os, c.avg_channel, h.avg_hour,
  a.cnt_app, d.cnt_device, o.cnt_os, c.cnt_channel, h.cnt_hour
FROM
  `talking.train_test2` as t
LEFT OUTER JOIN
  talking.mst_app_07 as a
ON
  a.app = t.app AND a.span = t.span
LEFT OUTER JOIN
  talking.mst_device_7 as d
ON
  d.device = t.device AND d.span = t.span
LEFT OUTER JOIN
  talking.mst_os_7 as o
ON
  o.os = t.os AND o.span = t.span
LEFT OUTER JOIN
  talking.mst_ch_7 as c
ON
  c.channel = t.channel AND c.span = t.span
LEFT OUTER JOIN
  talking.mst_ip_7 as i
ON
  i.ip = t.ip AND c.span = t.span
LEFT OUTER JOIN
  talking.mst_hour_7 as h
ON
  h.hour = t.hour
WHERE
  t.day = 8 AND t.span > 0

  SELECT
click_id,
ip,
app,
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
avg_day,
LAG(avg_day, 1) OVER(partition by ip order by click_time) avg_day_1,
LAG(avg_day, 2) OVER(partition by ip order by click_time) avg_day_2,
LAG(avg_day, 3) OVER(partition by ip order by click_time) avg_day_3,
LAG(avg_day, 4) OVER(partition by ip order by click_time) avg_day_4,
avg_hour,
LAG(avg_hour, 1) OVER(partition by ip order by click_time) avg_hour_1,
LAG(avg_hour, 2) OVER(partition by ip order by click_time) avg_hour_2,
LAG(avg_hour, 3) OVER(partition by ip order by click_time) avg_hour_3,
LAG(avg_hour, 4) OVER(partition by ip order by click_time) avg_hour_4,
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
sum_ip,
LAG(sum_ip, 1) OVER(partition by ip order by click_time) sum_ip_1,
LAG(sum_ip, 2) OVER(partition by ip order by click_time) sum_ip_2,
LAG(sum_ip, 3) OVER(partition by ip order by click_time) sum_ip_3,
LAG(sum_ip, 4) OVER(partition by ip order by click_time) sum_ip_4
FROM
`talking.train_test3`
WHERE
  click_id is null AND
  click_time <= '2017-11-08 16:00:00'


  SELECT
click_id,
ip,
sum_attr, last_attr, cnt_ip,
app,
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
avg_day,
LAG(avg_day, 1) OVER(partition by ip order by click_time) avg_day_1,
LAG(avg_day, 2) OVER(partition by ip order by click_time) avg_day_2,
LAG(avg_day, 3) OVER(partition by ip order by click_time) avg_day_3,
LAG(avg_day, 4) OVER(partition by ip order by click_time) avg_day_4,
avg_hour,
LAG(avg_hour, 1) OVER(partition by ip order by click_time) avg_hour_1,
LAG(avg_hour, 2) OVER(partition by ip order by click_time) avg_hour_2,
LAG(avg_hour, 3) OVER(partition by ip order by click_time) avg_hour_3,
LAG(avg_hour, 4) OVER(partition by ip order by click_time) avg_hour_4,
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
sum_ip,
LAG(sum_ip, 1) OVER(partition by ip order by click_time) sum_ip_1,
LAG(sum_ip, 2) OVER(partition by ip order by click_time) sum_ip_2,
LAG(sum_ip, 3) OVER(partition by ip order by click_time) sum_ip_3,
LAG(sum_ip, 4) OVER(partition by ip order by click_time) sum_ip_4
FROM
`talking.train_test3`
WHERE
  click_id is null AND
  click_time >= '2017-11-09 04:00:00' AND
  click_time <= '2017-11-09 15:00:00'


  SELECT
click_id,
ip,
sum_attr, last_attr, cnt_ip,
app,
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
avg_day,
LAG(avg_day, 1) OVER(partition by ip order by click_time) avg_day_1,
LAG(avg_day, 2) OVER(partition by ip order by click_time) avg_day_2,
LAG(avg_day, 3) OVER(partition by ip order by click_time) avg_day_3,
LAG(avg_day, 4) OVER(partition by ip order by click_time) avg_day_4,
avg_hour,
LAG(avg_hour, 1) OVER(partition by ip order by click_time) avg_hour_1,
LAG(avg_hour, 2) OVER(partition by ip order by click_time) avg_hour_2,
LAG(avg_hour, 3) OVER(partition by ip order by click_time) avg_hour_3,
LAG(avg_hour, 4) OVER(partition by ip order by click_time) avg_hour_4,
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
sum_ip,
LAG(sum_ip, 1) OVER(partition by ip order by click_time) sum_ip_1,
LAG(sum_ip, 2) OVER(partition by ip order by click_time) sum_ip_2,
LAG(sum_ip, 3) OVER(partition by ip order by click_time) sum_ip_3,
LAG(sum_ip, 4) OVER(partition by ip order by click_time) sum_ip_4
FROM
`talking.train_test3`
WHERE
click_id is not null
