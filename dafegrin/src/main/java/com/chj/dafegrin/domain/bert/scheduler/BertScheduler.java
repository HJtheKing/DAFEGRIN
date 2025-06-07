package com.chj.dafegrin.domain.bert.scheduler;

import com.chj.dafegrin.domain.bert.service.BertService;
import jakarta.transaction.Transactional;
import lombok.RequiredArgsConstructor;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Configuration;
import org.springframework.scheduling.annotation.Async;
import org.springframework.scheduling.annotation.EnableAsync;
import org.springframework.scheduling.annotation.EnableScheduling;
import org.springframework.scheduling.annotation.Scheduled;

@EnableScheduling
@EnableAsync
@RequiredArgsConstructor
@Configuration
public class BertScheduler {
    private final BertService bertService;

    @Value("${naver.news-keyword}")
    private String KEYWORD;
    @Value("${bert.url}")
    private String URL;

    @Async
    @Transactional
    @Scheduled(cron = "0 30 8 * * 1-5")
    public void getBert() {
        bertService.getBert(KEYWORD, URL);
    }

}
