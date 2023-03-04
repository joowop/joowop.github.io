---
title: "추천 시스템"
layout: archive
permalink: categories/recommend
author_profile: true
sidebar_main: true
---

{% assign posts = site.categories.recommender_system %}
{% for post in posts %} {% include archive-single2.html type=page.entries_layout %} {% endfor %}
