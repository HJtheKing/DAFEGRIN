package com.chj.dafegrin.domain.bert.dto;

import com.chj.dafegrin.entity.News;
import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Getter;
import lombok.Setter;

import java.time.LocalDateTime;

@Getter
@Setter
public class BertResult {
    private LocalDateTime date;
    private String content;
    private boolean label;
    @JsonProperty("positive_conf")
    private double positiveConf;

    public News toNews() {
        News news = new News();
        news.setDate(date);
        news.setContent(content);
        news.setPos(label);
        news.setPosProb(positiveConf);
        return news;
    }
}
