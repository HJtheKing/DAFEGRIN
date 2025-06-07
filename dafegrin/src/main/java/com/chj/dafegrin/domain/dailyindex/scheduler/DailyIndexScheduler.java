package com.chj.dafegrin.domain.dailyindex.scheduler;

import com.chj.dafegrin.domain.dailyindex.service.DailyIndexService;
import lombok.RequiredArgsConstructor;
import org.springframework.context.annotation.Configuration;
import org.springframework.scheduling.annotation.Async;
import org.springframework.scheduling.annotation.EnableAsync;
import org.springframework.scheduling.annotation.EnableScheduling;
import org.springframework.scheduling.annotation.Scheduled;

import java.time.LocalDate;
import java.time.ZoneId;
@EnableScheduling
@EnableAsync
@RequiredArgsConstructor
@Configuration
public class DailyIndexScheduler {

    private final DailyIndexService dailyIndexService;

    /**
     * 장 종료 후 코스피 지수 체크
     */
    @Async
    @Scheduled(cron = "0 0 21 * * 1-5")
    public void getKospi(){
        Double kospi = dailyIndexService.getKospi();
        LocalDate now = LocalDate.now(ZoneId.of("Asia/Seoul"));
        dailyIndexService.update(now, kospi);
    }
}
